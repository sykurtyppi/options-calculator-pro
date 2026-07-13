# Improvement Plan — Options Calculator Pro

_Authored 2026-07-12. Source: two independent audit passes (architecture/methodology + frontend/UX), cross-checked against the codebase. Every claim carries `file:line` evidence._

## Verdict

This is an unusually honest, well-engineered research codebase — the recent research-integrity PRs (#119–#122) are **real, not cosmetic**. Verified independently:

- Full suite: **1,435 pass / 0 fail / 0 skip**, fully offline (conftest blocks external IO by default).
- Crush-model CV is genuine OOF: `StratifiedGroupKFold` by symbol + `cross_val_predict`, scoring the deployed pipeline (`StandardScaler` + `CalibratedClassifierCV`), not bare in-sample LR — `institutional_ml_db.py:5345`.
- Train/serve skew killed via one shared transform — `services/crush_features.py`.
- Walk-forward leakage has a fail-loud sentinel — `structure_prior_store.py:553` `check_for_leakage`.
- Replay vs forward evidence strictly separated — `edge_engine.py:934`.
- Frontend ships a honesty regression suite — `web/frontend/src/honesty.test.js` (4/4).

## The unifying diagnosis

Both audits, run blind to each other, found the **same failure mode**: the honesty discipline is strong in the core and **breaks down at the seams** — data provenance is computed but neither persisted nor shown; simulated counts feed the live rank as if empirical; the view-model manufactures numbers the engine doesn't possess. Fixes cluster at persistence boundaries, ranking inputs, and the presentation layer — not in the core math.

---

## P0 — Honesty fixes at the seams (hours, highest payoff)

Small diffs that restore integrity the core already intends.

| ID | Fix | Evidence |
|----|-----|----------|
| **V1** (Critical) | Provider provenance erased at persistence. `option_source="provided"` is hardcoded and always truthy, so the ledger `or` never reaches the honest `marketdata_app`/`yfinance_fallback` label. A silent degrade to delayed yfinance is undetectable in the stored record. Prefer `metrics["data_sources"]["options_source"]`. | `earnings_vol_snapshot.py:714`, `recommendation_ledger.py:1014` |
| **FE-1** (High) | View-model manufactures a Best/Base/Worst percentage band from hard-coded multipliers (`ivShock*0.35`, `frictionStress*0.60`, `*0.75/*0.40/*0.35` haircuts) and renders it as a quantitative outcome distribution. Drop the numeric band; keep qualitative Fragile/Moderate/Robust. | `selectorViewModel.js:578-655`, `OutcomeRiskPanel.jsx:37-61` |
| **FE-2** (High) | "Priced move is cheap/rich" directional verdict fires on n≥1, contradicting the app's own n<8 reduced-evidence discipline. Gate on n≥6; neutral copy below. | `EarningsMoveHistoryPanel.jsx:39-45`, `earningsMoveHistory.js:48-52` |
| **V3** (High) | Simulated calendar counts feed the live selector rank labeled identically to realized evidence (`history_count = simulated_priceable_count`, weight 0.25). Add an `is_simulated` flag and propagate it. | `structure_scorecard.py:799-821`, `calendar_leg_picker.py:15` |
| **FE-4/5** (Med) | Big Decision-Quality % pill renders even on abstain (reads as "68% chance"); load-bearing honesty captions live only in `title=` tooltips (invisible to keyboard/touch/SR). De-emphasize pill on No-Trade, surface top captions as visible micro-text. | `SelectorDecisionCard.jsx:48-61` |
| **V6/V7** (Med) | `forward_screener.py:196` advertises the NBR model's "AUC=0.820 / 1,143 OOS" for a different classifier that actually reports OOF AUC 0.839 / n=1,639; `:228` `except: pass` silently voids the prediction. `screener_service.py:236/248/292/305` swallow earnings-anchor parse errors with no log. Correct citation + add logging. | `forward_screener.py:196,228`, `screener_service.py:236` |

---

## P1 — Validity & the measured/modeled boundary (days)

- **V2** (High): `_data_quality_score` and `diagnose_option_surface_quality` have no term for provider identity, greek availability, or quote delay — a yfinance surface (greeks all-NaN by construction, `yfinance_market_data_client.py:216`) can score `clean_surface`/`high` and pass the abstention gate. Add those terms. This is the backend counterpart to FE-3. — `earnings_vol_snapshot.py:1280`, `option_surface_quality.py:30`
- **FE-3**: Measured OOS/forward stats and score-derived tiles render in identical `.metric-card` chrome. Introduce a measured-vs-modeled visual token at the `Metric` level and apply consistently. — `DisplayAtoms.jsx:3-11`, `styles.css:452-484`
- **Leakage tests where the guarantee is only prose**: `scripts/build_earnings_iv_labels.py` (the primary label pipeline, "pre_event_date < earnings_date" at `:14`, as-of cutoff `:1122`) has **zero tests**; a future `trade_date` bleeding into a pre-event snapshot would poison every label and the suite stays green. Add explicit assertions. Same for `backtest_pre_earnings_otm_strangle.py` / `backtest_strategy.py` (leakage, not just contract-identity).
- **Abstain as first-class UI**: render `buildNoTradeSummary` reasons + "what would make this tradeable" on the Decision tab, not one click away in Evidence. — `DecisionEvidenceStrip.jsx:37-45`

---

## P2 — Architecture (weeks, structural leverage)

- **Break up the god-objects.** `InstitutionalMLDatabase` — ~5,100-line class, ~60 methods, 7 responsibilities (`institutional_ml_db.py:275`). Extract `OptionsPricingKit` (pure BS/IV, `:2379-2600`), `WalkForwardBacktester` (`:1075-3450`), `BacktestDiagnostics` (`:3839-4527`), `CrushModelTrainer` (`:5103-5411`). Then split `analyze_single_ticker` (918-line function, `edge_engine.py:2699-3617`) and fix the web-imports-CLI inversion (`app.py:483` imports `InstitutionalDataCollector` from `scripts/`). Relocate `edge_engine` out of `web/api/` so `scripts/*` stop importing the web layer.
- **Consolidate the 9 divergent release-timing normalizers** — a **correctness** fix. Inconsistent BMO/AMC cutoffs (`earnings_event_service.py:71` uses 9:30, `screener_engine.py:111` uses hour≥20/<16, `institutional_ml_db.py:1684` substring-only) mean the same event can be labeled differently by module. Collapse to one `services/release_timing.py`.

---

## P3 — Polish, a11y, hygiene (ongoing)

- **Charts have zero screen-reader support** (no `role="img"`/`aria-label`/data-table on any chart in `components/charts/`) yet carry the core evidence. Add takeaway `aria-label`s.
- **Finish the half-done color migration**: route `.tone-good/.tone-bad`, badges, quality-flags through `--pos/--neg/--warn`; swap legacy navy `rgba()` panels for `--panel`/`--surface-sunken`; fix two hardcoded chart-grid colors (`TermStructureChart.jsx:42,73`). — `styles.css:474-484,633-637,927-992`
- **Reorder the decision flow** so premise precedes conclusion: lift ticker input above screener/explainers; put a compact vol-state summary above the recommendation (currently one tab past it). — `App.jsx:460-508,640-648`
- **Complete the tab ARIA pattern** (tabpanel/aria-controls/roving arrow keys); bump `--muted-dim` (~3.9:1) to WCAG AA.
- **Hygiene**: delete ~18 squash-merged stale branches; clear ~188 MB stale `tmp/`+`exports/`+`reports/`.
- **Reconsider "institutional-grade" branding** (`institutional_ml_db.py:5`, `README.md:24`) — a single-researcher SQLite + yfinance pipeline calling itself institutional is the one certainty-claim the rest of the code scrupulously avoids.

---

## Sequencing

1. **P0 as one batch** — ~6 small, independent, high-integrity diffs; best effort-to-honesty ratio.
2. **P1 V2 + FE-3 as a pair** — the two ends of the same provenance hole.
3. **P2 god-object refactors after P0/P1 land** — splitting a 5,100-line class is safe only once provenance + label-leakage tests exist to catch regressions.

**Explicitly out of scope**: added model complexity (GJR/regime-switching) on a one-episode-dominated sample — it gives the pathology new hiding places for near-zero honesty gain.
