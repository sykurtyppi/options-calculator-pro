analyze_single_ticker — what I found

It's a 1,055-line orchestrator labeled legacy_orchestration_mode = "shared_snapshot_selector_adapter" — i.e. it explicitly exists to translate canonical VolSnapshot/SelectorOutput into a wide metrics dict for the UI. Internal phases:

1. Symbol clean / provider selection (2416-2423)
2. Price history (6mo + 5y) + market cap (2425-2501)
3. Option chain (MDA → yfinance fallback) (2504-2520)
4. Earnings resolution (MDA → shared resolver → yfinance) — three-layer fallback (2522-2621)
5. Build canonical snapshot, scorecards, selector (2623-2645) ← the real decision
6. Legacy term-structure extraction for charts only (2648-2664)
7. ~40 lines of _safe_float/getattr unpacking from snapshot_inputs (2666-2705)
8. ATM BSM Greeks (display only) (2707-2719)
9. Legacy historical move profile + expansion simulation (2721-2738)
10. Legacy edge math: raw_gross / confidence_adjusted / uncertainty / drawdown / expectancy (2740-2820)
11. Move-risk advisory (soft) (2822-2832)
12. Four calibration multipliers (ticker × kurtosis × crush × ML) (2834-2861)
13. Calendar + structure-specific payoff diagrams (2863-2944)
14. Hard gates (10 gates) (2946-2986)
15. Expansion gates (separate gate system) (2990-3064)
16. Mode selection (selector_first / iv_expansion / post_earnings_crush) + recommendation pull (3066-3086)
17. ~140-line rationale string array (3087-3231)
18. ~200-line metrics dict with ~150 keys (3233-3429)
19. EdgeSnapshot + ledger write (3431-3456)

Three significant smells in the orchestrator

1. Effectively-dead fallback branches. At lines 2820, 2861, and 3079 the code reads: confidence_pct = float(getattr(selector_output, "confidence_pct", <fallback>)). SelectorOutput always sets confidence_pct (it's a non-optional dataclass field), so every <fallback> branch is unreachable. The four-multiplier _calibration_mult (ticker × kurtosis × crush × ML) is computed, never used to set confidence_pct, but is reported in metrics. So the user sees confidence_calibration_mult=0.83 next to a confidence_pct that wasn't multiplied by it.
2. Two parallel gate systems (hard_gate_reasons + expansion_gate_reasons) with overlapping checks (spread, liquidity). They produce three "modes" gated by an if/elif chain at 3071-3077, and the chosen mode determines which gate list is "selected." Logic lives inline.
3. Legacy edge math is computed but ignored. Lines 2753-2820 compute raw_gross_edge_pct, expected_net_edge_pct, drawdown_risk_pct, expectancy_ratio — none of which feeds the recommendation (that comes from selector_output). They are reported in metrics for the UI. So the orchestrator does the math twice in different shapes.

---

Improvement plan — prioritized

Below is what I'd do, in execution order. Each phase is independently shippable; nothing requires the next phase to land first.

Phase 1 — Leakage gating on priors and calibration (highest value, highest risk)

The single correctness fix the system needs.

1.1 Add per-observation timestamps to StructurePriorStore

Today the JSON keeps running sums (sum_return_pct, positive_count, observation_count) per structure. That makes as-of filtering impossible without losing the data.

Change: keep a per-structure observations: list[{date, source_type, realized_return_pct, realized_expansion_pct, observation_id}] array. Derive win_rate, avg_return_pct, rank_score on read, parameterized by as_of_date and optionally source_types filter. Keep the existing aggregate fields as a denormalized cache for the no-filter path so the production read stays O(1).

Migration: if the existing JSON has only the aggregate, treat its last_updated date as the timestamp for all its accumulated observations (preserves current behavior). Bump schema_version to 2. Write a one-time backfill that rebuilds the observation list from the outcome_trades SQLite table — that table already has entry_date, exit_date, source_type, realized_return_pct, and realized_expansion_pct, so the data is fully recoverable.

1.2 Same shape for IVExpansionCalibration

It already stores parallel lists _scores, _expansions, _sources, _observation_ids. Add a parallel _dates list and persist it. The isotonic fit already iterates the parallel lists, so adding a date filter is one zip call. Same migration story (rebuild from outcome_trades).

1.3 Plumb as_of_date through the read path

- StructurePriorStore.get_prior_dict(structure, as_of_date=None) and load_all_structure_priors(as_of_date=None).
- IVExpansionCalibration.apply(score, as_of_date=None).
- structure_scorecard._load_walk_forward_priors(as_of_date=None) and build_structure_scorecards(snapshot, as_of_date=None).
- Callers pass snapshot.as_of_date. Production callers can pass None to mean "now" and keep current behavior.
- The mtime-based cache signature in _walk_forward_prior_signature has to extend to include as_of_date in the cache key, or be replaced with an LRU keyed on (signature, as_of_date). Otherwise as-of calls compete with each other.

1.4 Per-run store isolation for seeding/backtest

scripts/seed_outcomes_from_replay.py currently defaults prior_store_path=None → production path. Fix:
- Default the seeding paths to None → temp paths under tmp/seed_run_<utc>/, not production. Require an explicit --target=production to write to the live store.
- Add the same flag to run_replay_backtest.py and any other script that calls finalize_trade_and_update_learning.
- The _DEFAULT_STORE constants in structure_prior_store.py and calibration_service.py should be set via env var (OPTIONS_CALCULATOR_PRIORS_PATH, OPTIONS_CALCULATOR_CALIBRATION_PATH) so a backtest harness can override the singleton without code changes.

1.5 Lock down the read side

In build_structure_scorecards, when called with an as_of_date that is in the past (i.e., not "now"), fail loudly if the persistent prior store contains observations with date > as_of_date and source_type != "replay". That's a sentinel for "you've leaked real paper/live trades into a backtest." Replays are fine to bypass because they're the synthetic backtest signal.

1.6 Tests that make leakage failures observable

This is the property that doesn't exist anywhere in the test tree:

- test_priors_respect_as_of(tmp_path): write 3 observations dated 2024-01-01..03 and 3 dated 2025-01-01..03. Call get_prior_dict(as_of_date=2024-06-01). Assert N=3 and that the win-rate reflects only the first three.
- test_calibration_respects_as_of: same.
- test_scorecard_backtest_isolation: seed observations into a temp store, build a scorecard for an earlier as_of_date, assert the rank_score matches a pre-seed baseline.
- test_seed_does_not_write_to_production (regression): mock _DEFAULT_STORE to a path under tmp, run seed with default args, assert the production path was never touched.

These four tests alone would have caught the issue and will catch the regressions.

Why this is Phase 1: the on-disk store has only 5 observations on otm_strangle right now. Adding timestamp granularity and as_of plumbing is cheap now and gets exponentially harder once the stores grow. Do it before the next forward-loop run accumulates more state.

---

Phase 2 — Coverage gaps that map 1:1 to the model-card claims

2.1 tests/unit/test_services/test_structure_prior_store.py (new file)

The central learning-loop state has no dedicated test file. Minimum surface:

- Laplace smoothing crossover at LAPLACE_SMOOTHING_THRESHOLD=10 (one observation below, one above)
- MIN_OBS_FOR_OVERRIDE=5 boundary (4 obs → None, 5 obs → dict)
- source_types accumulator correctness for replay/paper/live combinations
- Round-trip save/load preserves all fields
- Concurrent writes don't corrupt the JSON (use threading.Thread × 16, assert final count == sum of writes)
- Unknown structure name is silently dropped (current behavior — pin it)

2.2 Round-trip integration tests for the learning loop

The finalize → priors-update → scorecard-sees-update flow has no test. Two tests:

- test_finalize_updates_priors_and_invalidates_cache: assert the cache signature changes and the next build_structure_scorecards reflects the new win_rate. (Today: this property is what causes leakage; tomorrow once as-of gating exists, the test is "production calls see the update; backtest calls at an earlier as-of don't.")
- test_calibration_observation_dedup_via_observation_id: finalize the same trade_id twice, assert N increments by 1.

2.3 Model-card-claim audit

Pick the four cards with the loudest claims (structure_scorecard, structure_selector, recommendation_ledger, forward_performance_diagnostics) and write a tests/unit/test_model_card_claims/test_<name>_claims.py per card. Each test function name matches one bullet from the card's "Validation Approach" section. This makes drift visible: if the card claims "tests cover stale-source comparison" and there's no test by that name, CI fails.

Mechanical, but it's the test of whether the disclosure is true, which is the part the system most needs to keep honest.

2.4 Property tests for the no-trade gates

structure_selector has six gate conditions for RECOMMENDATION_NO_TRADE at lines 126-132 of structure_selector.py. Each combination should have a test that toggles exactly one condition and asserts the recommendation flips. This catches "I added a 7th condition and forgot to lower the threshold on the 4th" — the exact class of bug your recent P-5c commits are tuning around.

---

Phase 3 — edge_engine.py decomposition

3,456 lines → ~7 modules. The split is mechanical (no logic changes) and unlocks the orchestrator cleanup in Phase 4. Order matters: do the leaves first so the orchestrator changes can be small and reviewable.

3.1 Move pure-math leaves first (no I/O, no provider coupling)

Create a package web/api/edge_engine/ with __init__.py re-exporting analyze_single_ticker and EdgeSnapshot for backwards compat:

- pricing.py (lines 1686-2197 → ~510 lines): _bsm_greeks, _derive_iv_scenarios, _calendar_spread_payoff, _straddle_payoff, _strangle_payoff. Pure functions; trivial to test in isolation.
- stats.py (lines 2198-2333 → ~135 lines): _rv_percentile_and_regime, _excess_kurtosis, _kurtosis_confidence_mult, _classify_ticker_tier, _crush_calibration_mult.
- historical_moves.py (lines 1541-1685 → ~145 lines): _historical_earnings_move_profile.
- ml_crush.py (lines 60-323 → ~260 lines): all three crush-model functions.

These four pull ~1,050 lines out with zero refactoring risk because nothing imports private leaf functions from edge_engine.

3.2 Then the provider-flavored extractors

- mda_extract.py (lines 1008-1324 → ~315 lines): MarketData term/smile/earnings extraction.
- yf_extract.py (lines 1325-1540 → ~215 lines): yfinance counterparts.

While doing this, define a Provider Protocol with three methods (term_structure(...), smile_curvature(...), next_earnings(...)). MDA and yfinance both implement it. The orchestrator then has provider = mda_provider if use_mda else yf_provider once, instead of branching three times. This unblocks orchestrator testability — you can pass a fake provider for unit tests.

3.3 Then the calendar simulator

- calendar_sim.py (lines 433-748 → ~315 lines): _select_pre_earnings_calendar_contracts, _lookup_exact_contract_row, _simulate_pre_earnings_calendar_trade, _summarize_pre_earnings_expansion. Coupled to the options_feature_store but otherwise standalone.

After this step edge_engine.py is ~1,500 lines and almost entirely the orchestrator plus EdgeSnapshot.

---

Phase 4 — Decompose analyze_single_ticker

Now the orchestrator is the only thing left in edge_engine.py worth changing. Target shape: ~150 lines total, each step a named function.

4.1 Extract the build-the-inputs phase (lines 2425-2645 → inputs.py)

A single function build_analysis_inputs(symbol, mda_client) -> AnalysisInputs returning a frozen dataclass with: current_price, close, close_for_profile, hist_long, market_cap, ticker_tier, chain_df, dte, earnings_release_time, earnings_events_for_profile, earnings_metadata, vol_snapshot, structure_scorecards, selector_output, data_source, options_source, provider_telemetry_summary.

That removes 220 lines from the orchestrator and gives you a clean seam for "give me the snapshot and selector output for an as-of date" — which is exactly what a backtest harness wants, and what doesn't exist today.

4.2 Delete the dead legacy edge math (lines 2753-2820)

raw_gross_edge_pct, expected_gross_edge_pct, expected_net_edge_pct, expectancy_ratio, expectancy_score — these duplicate what's in lead_scorecard.expected_edge_pct / lead_scorecard.composite_structure_score. They're reported in metrics for the UI but they're computed from the same underlying snapshot inputs the scorecard uses, so they should be derived in the scorecard, not the orchestrator.

Two options:
- (preferred) Move them into StructureScorecard as additional fields and have analyze_single_ticker just read them off the lead scorecard. Single source of truth.
- (smaller) Extract them into legacy_edge_math.py and call it. Keeps the duplication but isolates it.

Go with option 1 — the values are already conceptually scorecard outputs.

4.3 Collapse the dead confidence-multiplier branch

Lines 2820, 2861, 3079 all do getattr(selector_output, "confidence_pct", <unreachable_fallback>). Pick one:
- Decision: if the four multipliers (ticker_tier × kurtosis × crush × ML) are still wanted, apply them inside the scorecard or selector — wherever the canonical confidence is computed. Then they're real.
- Or: if they're informational only, rename them in the metrics dict from confidence_calibration_mult to confidence_calibration_mult_informational and stop multiplying them into confidence_pct_raw (which itself is dead).

Today the UI sees a multiplier that doesn't actually multiply anything. Pick one of those two states.

4.4 Promote the gate system into its own module (gates.py)

The two gate lists (hard_gate_reasons, expansion_gate_reasons) are ~80 lines of inline if-chains. Move them to a gates.py with two functions: evaluate_hard_gates(snapshot, inputs) -> list[GateResult] and evaluate_expansion_gates(snapshot, inputs, expansion_summary) -> list[GateResult]. GateResult is {name, passed, reason, threshold, observed}. Now each gate is independently testable and the metrics dict can serialize gates structurally instead of as flat strings.

4.5 Extract rationale and metrics builders

- rationale.py: build_rationale(snapshot, selector_output, gates, ...) -> list[str]. 140 lines, pure formatting.
- metrics_builder.py: build_metrics(snapshot, selector_output, scorecards, inputs, gates, payoffs, ...) -> dict. 200 lines, pure assembly.

After these extractions analyze_single_ticker is roughly:
def analyze_single_ticker(symbol, mda_client=None, record_to_ledger=True):
    inputs = build_analysis_inputs(symbol, mda_client)
    expansion = summarize_pre_earnings_expansion(...)
    payoffs = build_payoff_diagrams(inputs)
    gates = evaluate_all_gates(inputs, expansion)
    mode = pick_analysis_mode(inputs.selector_output, gates, expansion)
    rationale = build_rationale(inputs, gates, expansion, mode)
    metrics = build_metrics(inputs, payoffs, gates, expansion, mode)
    return finalize_snapshot(symbol, inputs, metrics, rationale, record_to_ledger)
~30 lines.

---

Phase 5 — Things I'd defer

Not because they don't matter, but because they're cheaper after Phases 1-4:

- Splitting web/api/app.py (1,698 lines). Standard FastAPI app — split by router group. Easy, but no correctness payoff.
- Reducing the metrics dict to ~50 keys. The 150-key dict is a UI contract; shrinking it requires touching the frontend.
- Replacing getattr(selector_output, X, default) with attribute access. Mechanical cleanup; do it as part of Phase 4.4/4.5 if it's the same file.

---

Sequencing and risk

Phase: 1.1-1.5 (gating)
Files touched: 5
Lines moved/changed: ~400 + schema migration
Risk: Medium — data migration
Why this order: The only correctness issue; cheap before stores grow
────────────────────────────────────────
Phase: 1.6 (leakage tests)
Files touched: new
Lines moved/changed: ~200
Risk: Low
Why this order: Pins the invariant for everything after
────────────────────────────────────────
Phase: 2.x (coverage)
Files touched: new
Lines moved/changed: ~600
Risk: Low
Why this order: Independent; can run in parallel with anything
────────────────────────────────────────
Phase: 3.1-3.3 (mechanical splits)
Files touched: ~10
Lines moved/changed: ~1,800 moved
Risk: Low
Why this order: No logic change; reviewable as pure refactor
────────────────────────────────────────
Phase: 4.1-4.5 (orchestrator)
Files touched: ~6
Lines moved/changed: ~700 changed
Risk: Medium
Why this order: Touches recommendation surface; needs Phase 2 tests in place first

Phase 1 + Phase 2.1-2.2 are the high-value pair to do together. After that you have an honest backtest framework, the central state has a test file, and the leakage invariant is pinned. Phase 3 is a calm refactor you can do over a few sittings. Phase 4 is the one that benefits most from doing 1-3 first — the orchestrator's seams become obvious once the provider extractors are a Protocol and the scorecard owns its own edge math.

Want me to start on any of these? Phase 1.1+1.2+1.3 (the gating fix, no migration yet) is the smallest change that closes the correctness gap; Phase 3.1 (move the four pure-math leaves out of edge_engine.py) is the smallest change that makes the file readable.
