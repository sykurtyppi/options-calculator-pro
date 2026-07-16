# AGENTS.md — options-calculator-pro

Per-repo facts for the Hermes review agent (adversarial quant reviewer). This file is the
repo-specific layer under the global SOUL.md identity and the shared six-check review skill
(look-ahead bias, point-in-time violations, survivorship, overfitting/multiple-testing,
cost/execution realism, regime cherry-picking). Report findings as a severity-ranked list
citing exact lines — never a prose summary.

> **Audit provenance.** Verified against `main` at commit `f85cada` on 2026-07-16. Package
> versions, line numbers, sample counts, and "currently surviving" observations are
> point-in-time snapshots — re-verify any specific line / number / version against the
> current tree before relying on it. The structural invariants (data providers, risk tier,
> the classes of pitfall) are durable; the exact citations are not.

## Orientation

Pre-earnings **IV-crush volatility arbitrage** research system, primarily ATM **calendar spreads**
(short front-month / long back-month), with straddle/strangle variants. Core signal is
**NBR = front_iv / back_iv** (`trading_system/signal_engine.py`). Timing: screen at T-1, enter T-2,
exit T+1. It is a research + **forward-paper** system — no live capital, no broker. The whole point of
review here is that the strategy's edge is dominated by data-integrity and multiple-testing questions.

## 1. Stack & dependencies

- **Python 3.11–3.12** (`.python-version`, `pyproject.toml`). `numpy==2.4.4`, `scipy==1.17.1`,
  `pandas==2.3.3`, `scikit-learn==1.8.0`, `duckdb==1.5.2`, `yfinance==1.3.0`. **No statsmodels,
  no QuantLib.** Options math is hand-rolled Black-Scholes-Merton (`web/api/edge_math.py`); ML is
  sklearn only. Frontend: React + Vite (`web/frontend/`).
- **Entry points**: FastAPI backend `web/api/app.py`; ~46 CLI `scripts/` are the research/backtest/
  train entrypoints; always-on evidence loop via **macOS launchd** plists (`scripts/automation/*.plist`:
  evidence-cycle, forward-paper-collector, candidate-exit-resolver, weekly-evidence-report, watchdog).
  This is an evidence-collection loop, **not** a trading loop.
- **Data stores**: SQLite `~/.options_calculator_pro/institutional_ml.db`
  (`services/institutional_ml_db.py`); **Parquet EOD option chains** on external drive
  `/Volumes/T9/market_data/research/options_features_eod/`; JSONL ledger/outcome stores; model `.pkl`
  in `~/.options_calculator_pro/models/`.
- Tests run offline — `services/external_io_gate.py` + `conftest.py` enforce no network. Respect that
  guard if touching tests.

## 2. Risk tier — Research/backtesting (forward-paper only)

**No broker integration found** — grep for `place_order|alpaca|ibkr|tastytrade` returns zero matches (strong evidence, not exhaustive proof).
`docs/architecture.md`: "This is not an execution engine, broker, or financial adviser." Forward
"trading" is **paper only** (`scripts/forward_paper_collector.py`, `candidate_shadow_outcome.py`);
without `--forward-trade-log-path` the tracker status is `simulated_backtest`. Position sizing is a
fixed contract count. Therefore the **primary risk is overfitting / multiple-testing**, and for any new
result the review question is: **is there genuine out-of-sample / walk-forward / replay validation, or
is this in-sample discovery dressed as a result?** All signal/pricing/sizing/backtest findings are
**flag-only**.

## 3. Known pitfalls specific to this repo (verified from code)

1. **Two backtest engines — one synthesizes P&L.** The **real-price** path
   (`scripts/backtest_strategy.py`, `scripts/optimize_entry_exit.py`) uses actual entry/exit mid prices
   from T9 parquet chains; cost = `(total_spread/2) × 0.75 × 100` per contract (`FILL_REALISTIC=0.75`),
   **no commissions/fees/market-impact**. The **synthetic** path
   (`institutional_ml_db.py::_simulate_walk_forward_trade`) **prices no option** — `gross_return_pct` is a
   heuristic of realized-vol/momentum features (`theta_carry + stability_bonus + regime_bonus + … −
   penalties`), with proxied liquidity (`option_volume_proxy = sqrt(share_volume)×0.5`). Docs concede it is
   "a proxy engine … treat outputs as ranking guidance, not execution expectancy." **On every result, force
   the distinction: real-price fills vs synthetic feature-derived P&L.** `scripts/run_replay_backtest.py`
   runs a hybrid that can silently mix the two.
2. **Earnings-date look-ahead — the crux.** `services/earnings_event_service.py` resolves dates from
   AlphaVantage/FMP/SEC/yfinance as **current (non-point-in-time) values**, and
   `institutional_ml_db._build_proxy_earnings_schedule` **synthesizes a ~63-trading-day quarterly schedule
   with a per-symbol hash phase** when true dates are missing (`source='proxy'`, `release_timing='UNKNOWN'`;
   enabled by default in backfill). **Treat any `source='proxy'` event as non-evidence.** BMO/AMC timing is
   admitted unreliable and **normalized inconsistently across 9 code sites** (`docs/IMPROVEMENT_PLAN.md`),
   which misaligns the T-2/T-1/T+1 windows.
3. **IV point-in-time integrity gaps (admitted).** (a) provider provenance can be silently erased at
   persistence (`option_source="provided"` hardcoded, so a degrade to delayed yfinance is invisible in the
   stored record — `IMPROVEMENT_PLAN.md`); (b) the primary label pipeline `scripts/build_earnings_iv_labels.py`
   enforces `pre_event_date < earnings_date` in prose but has **zero leakage tests** — a future `trade_date`
   bleeding into a pre-event snapshot would poison every label and the suite stays green.
4. **Survivorship / look-ahead membership.** `INSTITUTIONAL_UNIVERSE` (`institutional_ml_db.py:43-46`) is the
   **current ~504-name S&P 500 list** ("Tier 2 added 2026-03-01"); backtests iterate this static current
   membership, so delisted/removed names are absent and recently-added names are backfilled into the past.
   `signal_engine.SECTOR_MAP` is likewise current-membership.
5. **Multiple-testing is real and documented.** `scripts/optimize_entry_exit.py` sweeps entry ∈
   {T-7…T-1} × exit ∈ {T+0,T+1} (10 combos, picks best); `run_backtest_parameter_sweep` grids crush-gate
   thresholds/hold days → `_best_params.json`. `docs/CALENDAR_PICKER_PROMOTION_2026-05-27.md` states the
   `candidate_min_dte` rule was **discovered by sweeping** 10 symbols × 139 events × 3 horizons × 4 structures
   and picking the best in-sample config, and names the trap itself ("textbook multiple-testing failure mode,
   Harvey & Liu"). **Treat sweep outputs as in-sample discovery, not validation.**
6. **Genuine OOS/adversarial harness exists — credit it, then check it's actually run.** Walk-forward NBR
   thresholds use prior years only (`backtest_strategy.py`); rolling OOS (`--oos-validation`);
   `scripts/adversarial_stress_test.py` does permutation/placebo (shuffle targets within year — failure to
   collapse flags leakage), leave-one-symbol-out, and per-year regime stability; `structure_prior_store.py::
   check_for_leakage` is an as-of sentinel. **But** the governing docs' own verdict is "the system is
   collecting evidence, not yet validating performance" — forward-OOS promotion needs ≥40 distinct events
   accumulated *after* the PR-AE merge, a clock that had just started.
7. **Cost realism is optimistic.** Paper profile `commission_per_contract=0.0`
   (`services/execution_cost_model.py`); real-price backtest omits commissions; slippage is a fraction of
   quoted spread with no adverse-selection/partial-fill modeling. The `EXECUTION_COST_DOMINATES_EDGE_RATIO=0.50`
   veto operates on *scorecard* cost, not realized fills.
8. **Theoretical calendar pricing.** Calendars priced off interpolated IV30/IV45 with a fabricated back-expiry
   (near+28d), flagged `calendar_is_theoretical: True`, "not guaranteed to match a live quoted chain"
   (`edge_math.py`). `services/iv_term_structure.py::bounded_interp` correctly refuses to extrapolate IV
   (a prior bug fabricated an iv30 that fed a 32%-weight ranking score).
9. **"Monte Carlo" is scenario derivation, not path simulation.** `edge_math.py::_derive_iv_scenarios` applies
   post-earnings IV scenarios (expand/flat/crush) from the historical move distribution; docs state "no random
   draws are used in the backtest engine." `np.random` appears only in permutation tests and backfill.
10. **Simulated counts ranked as empirical / prior contamination.** `history_count = simulated_priceable_count`
    feeds the selector rank (weight 0.25) "labeled identically to realized evidence" (`IMPROVEMENT_PLAN.md`);
    PR #71 fixed a straddle/strangle prior inflated ~37× by cartesian multi-counting (`docs/PR71_…`). Treat any
    `simulated_*` count as non-evidence.

## 4. What the agent may fix directly vs only flag

**Default posture is read-only.** During a review-only task, report proposed changes as
findings and do not edit; post inline PR comments only as the configured review bot or when
explicitly asked, not merely because a PR exists. Fixes apply only when explicitly
authorized — and even then, numerical / signal / statistical changes require focused
before/after validation and human review, never a silent edit. "Low-risk" is not risk-free:
UI, scheduler, deploy, CORS, and DB code can still be consequential — treat every item below
as a candidate, not standing authorization.


**Flag only — never auto-fix (signal generation / pricing / sizing / backtest / labels):**
`web/api/edge_math.py`, `web/api/edge_engine.py`, `services/institutional_ml_db.py` (all backtest/simulator/
crush/earnings-resolver logic), `trading_system/signal_engine.py`, `services/earnings_event_service.py`,
`services/earnings_vol_snapshot.py`, `services/iv_term_structure.py`, `services/structure_selector.py`,
`services/structure_scorecard.py`, `services/execution_cost_model.py`, `services/execution_scenarios.py`,
`services/candidate_exit_resolver.py`, `services/structure_prior_store.py`, `services/realized_vol.py`, and
every `scripts/backtest_*`, `scripts/optimize_*`, `scripts/train_*`, `scripts/build_earnings_iv_labels.py`,
`scripts/*replay*`, `scripts/adversarial_stress_test.py`.

**Low-risk — only if a fix is explicitly requested, (style/infra, no signal impact):**
`utils/logger.py`, `services/jsonl_helpers.py` / `sqlite_helpers.py`, `provider_telemetry.py`,
`scripts/rotate_launchd_logs.py`, `scripts/automation/*` (plists, shell wrappers, installers),
`.github/workflows/ci.yml`, `conftest.py`/test scaffolding (respect the offline gate),
`web/frontend/` styling/formatters, docstrings/typing/imports.

## 5. PR etiquette

Findings as **inline review comments on exact lines**, severity-ranked, each stating the concrete way the
number could be wrong (real-vs-synthetic P&L, proxy earnings, as-of leakage, survivorship, in-sample sweep,
zero-commission cost). No full-file rewrites unless explicitly asked to push a fix commit. For flag-only
areas, comment and stop.
