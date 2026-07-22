# Deep-Dive Fix Plan — Earnings-Move Integrity & Event-Vol Validity (DD series)

Status: DD-1 + DD-2 IMPLEMENTED (PR-A, 2026-07-22) after independent
adversarial review (all load-bearing citations verified; decisions below
marked RESOLVED). Verified live: PYPL median 8.62 (was 2.19), p90 13.88,
2022-02-01 event 24.6% (was 2.24%), event decomposition honestly suppressed
(status event_expiry_not_quotable — the 9-DTE spanning expiry exists but
fails the yfinance spread bar). PR-B = DD-4 (regime-aware entry score)
remains TODO, separately, with a full before/after ranking diff. DD-5 copy
notes folded into PR-A's UI changes.

REVIEW-RESOLVED DECISIONS:
- Q1 RESOLVED: unknown-timing events are DROPPED from measurement + counted
  (`unknown_timing_event_count`), NOT bracket-measured (max-of-two-windows is
  a positively-biased estimator; winsor at n≈20 doesn't clip it; post-2a the
  unknown residue is small; n<5 falls to the honest daily_fallback path which
  the scorecard already sentinels at 0.15).
- Q2 RESOLVED: spanning rule is timing-aware: reaction_session = earnings
  date D for BMO/intraday; D+1 for AMC; D+1 (conservative) for unknown.
  Spanning iff expiry_date >= reaction_session. Do NOT reuse
  `_pick_front_expiry_after` verbatim (it is strictly-after and would skip a
  valid BMO same-day expiry).
- Q3 RESOLVED: DD-4 uses a percentile-conditioned cap
  (`rv_percentile_rank >= 90 → entry input = max(iv_rv_trailing, iv_rv_har)`),
  not a global α-blend. Separate PR with explicit ranking diff.
- Q4 RESOLVED: one PR for DD-1+DD-2 (single golden regeneration), two
  commits for bisectability. DD-4 separate. DD-5 folded into PR-A.
- D1: pin timezone before hour extraction — preserve exchange wall-clock
  (tz-aware → convert to America/New_York, then drop tz; naive → as-is).
- D2 (documented residual): nonzero-but-wrong Yahoo placeholder stamps
  (e.g. 10:00 on a true AMC) classify confidently wrong and no backstop can
  catch them. Accepted residual risk; noted in code comment.
- D3: when event decomposition is unavailable, the scorecard must degrade
  LOUDLY — sentinel the move-fit component (mirror the daily_fallback 0.15
  pattern at structure_scorecard.py:232), not silent neutral 0.50s.
- D4: locate the event-spanning expiry in the FULL pre-cap expiry set (line
  877 set / chain_frame), not the max_term_expiries=6-capped expiry_stats.
- D5: the event-spanning expiry must pass the same spread-quality bar as the
  front leg before its implied move feeds decompose_event_vol; else status
  = no_event_spanning_expiry (quotable).
- Missing#3: dropped unknown-timing events STILL contaminate the RV baseline;
  `_event_contaminated_session_dates` excludes BOTH candidate sessions (D and
  D+1) for unknown-timing events.
- Missing#4: verify historical_vs_implied and tail_vs_implied ratios derive
  from the decomposition output (single source) so historical-vs-implied is a
  same-expiry comparison.
- Missing#1: snapshot to_dict/schema-shape tests join the golden-regeneration
  list for the new fields.
Origin: PYPL deep-dive 2026-07-22 — user-reported "mixed readings" between the
ranked screener (#1, setup 0.85) and the full edge analysis (No Trade, −5.1%
modeled edge, MOVE RISK ELEVATED banner). Root-cause investigation confirmed
two real bugs, one regime artifact, and one presentation mismatch.

---

## 0. Confirmed findings (all verified live against the running server)

| ID | Severity | Finding |
|----|----------|---------|
| DD-1 | **High** | Historical earnings-move profile mismeasured when release timing is unknown: served PYPL median **2.19%** vs true **8.62%** (p90 9.1 vs 13.9). Live path: yfinance provider client sets `reportTime=None` for ALL historical rows → `"unknown"` → shared profile silently measures with the **BMO window** → AMC-era reactions land on the wrong day (e.g. 2022-02-01 shown as 2.24%; the real reaction was ≈−25% the next session). |
| DD-2 | **High** | Event-vol decomposition runs on a near-term expiry that does **not span the earnings date** (PYPL: 2-DTE 7/24 expiry vs 7/28 BMO earnings). `event_implied_move_pct` (≈3.2%) is fabricated from an option containing zero event variance; `move_risk_ratio` (p90/event-implied ≈2.9, "ELEVATED") is a category error. The legacy hard gate detects the condition but only gates the trade — headline metrics still display. |
| DD-3 | Medium | `implied_move_pct` always uses the nearest expiry regardless of event coverage, and the UI presents it beside earnings-move stats, inviting an invalid comparison. (Presentation face of DD-2.) |
| DD-4 | Medium | Ranked-screener entry score treats trailing `iv_rv_ratio = 0.80` as a **perfect 1.00** long-vol entry, but trailing RV is at its 100th percentile (regime-inflated). Forward-looking `iv_rv_har = 1.42` says vol is **rich**. The snapshot's own `decomp_regime_warning` fires, yet ranking ignores it. PYPL's #1 rank is largely this artifact. |
| DD-5 | Info | The 18.9% near-term spread (yfinance delayed quotes) drives the all-structures-ineligible abstention. This is the H2/H7 data-quality design working as intended — **no change**, but the two-verdicts-one-page presentation (screener says #1, analyzer says No Trade) deserves an explicit cross-reference note. |

Empirical validation already done:
- yfinance historical index timestamps carry real report times for PYPL
  (16:00/17:00 ET = AMC through 2024-02; 07:00/08:00 ET = BMO from 2024-04),
  and `normalize_release_timing` classifies all 24 correctly when the time is
  preserved. The DD-1 source-fidelity fix is therefore viable.
- Fresh in-process runs with timing preserved produce median 8.62 / p90 13.88 /
  avg4 10.16 (n=20) for PYPL — the numbers the fix should reproduce end-to-end.

---

## 1. Architecture principles (consistent with the safe_mid saga, PRs #126–#129)

1. **Fix the invariant once, at the narrowest shared seam** — not per-call-site.
   DD-1's seam is `compute_earnings_move_profile` (measurement safety) plus the
   provider clients (source fidelity). DD-2's seam is the snapshot-inputs layer
   where the decomposition is invoked (validity guard).
2. **Unknown must degrade loudly, never silently guess.** "unknown" timing may
   not silently pick the BMO window; a non-spanning expiry may not silently
   produce an "event" decomposition. Each degradation must surface as a flag
   that reaches the metrics payload (and therefore the UI).
3. **One number, one answer.** The screener and the analyzer disagreeing on
   "median earnings move" for the same symbol (8.6 vs 2.19) is itself a defect
   class. The fix must make both surfaces converge on the same profile given
   the same inputs; a regression test should pin that convergence.

---

## 2. DD-1 design — timing-safe earnings-move profile

Two independent layers, both required (defense in depth):

### 2a. Source fidelity (yfinance provider client)
`services/yfinance_market_data_client.py::get_earnings` currently discards the
index timestamp's time-of-day (`idx.date()`) and pins `reportTime=None` for
historical rows. Change: derive `reportTime` for historical rows from the raw
index timestamp via the shared `normalize_release_timing` (16:00 → AMC,
07:00 → BMO, midnight/dateonly → None as today). The docstring's claim that
downstream treats None safely is false today and becomes true via 2b.

### 2b. Measurement safety (shared profile)
`services/earnings_move_profile.py::compute_earnings_move_profile` — for
events whose timing is `"unknown"`, replace the silent BMO assumption with a
**timing-agnostic bracket window**: move = max-magnitude single-session move
within {pre→event, event→next} — i.e. measure both candidate windows and take
the one with the larger |move|.
  - Rationale: for a true BMO event the event-day move dominates; for a true
    AMC event the next-day move dominates. Picking the larger recovers the
    reaction day in both cases. Upward bias exists (max of two draws) but is
    bounded by one ordinary session's noise and is far smaller than the
    current downward corruption (which replaces the reaction with noise).
  - Every unknown-timing event measured this way is counted; the profile gains
    `unknown_timing_event_count` so consumers can see sample degradation.
  - Known-timing events (BMO/AMC/during) keep today's exact windows —
    no behavior change on the healthy path.

### 2c. Degradation flag propagation
`EarningsMoveProfile` gains `unknown_timing_event_count: int`. The vol
snapshot and edge metrics surface it (e.g. `earnings_move_unknown_timing_count`)
so the "Last N earnings" panel can annotate events measured under the bracket
convention. Frontend: annotate rather than redesign (small caption on the
history panel when count > 0).

### 2d. Convergence (research-informed)
Research confirmed the timing-correct logic already exists **three times**:
- `services/screener_service.py:286-321` `_collect_yf_past_earnings_events`
  (hour thresholds <9:30/>=16) — the reason the screener shows 8.6%.
- `web/api/edge_engine.py:2599-2635` fallback Paths B/C (correct, but dead in
  production because the lossy Path A always populates first at :2597).
- `scripts/build_earnings_iv_labels.py:417-438` (label pipeline; NOTE: uses
  divergent thresholds >=15/<=10 — out of scope here, logged as follow-up).

Therefore 2a is not new logic: `yfinance_market_data_client.get_earnings`
(:277 `idx.date()` discards the signal; :282 pins None) should feed the full
index Timestamp through the CANONICAL `normalize_release_timing`
(services/earnings_move_profile.py:29) — same thresholds as the screener path.
Convergence test: same synthetic AMC-era history through the edge adapter
(`edge_engine._historical_earnings_move_profile`) and the snapshot adapter
must produce equal medians (extends the existing 8-decimal reconciliation
test at tests/unit/test_services/test_earnings_vol_snapshot.py:397-419).

### 2e. Consistency obligation (from research)
`services/earnings_vol_snapshot.py:1154-1194` `_event_contaminated_session_dates`
hardcodes the same AMC-shifts-next-session convention for RV-baseline
exclusion. Any change to unknown-timing measurement (2b) must be mirrored
there (exclude BOTH bracket sessions for unknown-timing events) or the RV
baseline and the move measurement diverge on the same event.

### 2f. Tests that pin the bug (must change with the fix)
- `tests/unit/test_services/test_yfinance_market_data_client.py:137` asserts
  `df["reportTime"].isna().all()` for historical rows — locks in the bug; the
  fixture (:50-56) uses midnight timestamps and cannot represent AMC. Both the
  assertion and fixture change; add a red-then-green case with 16:05/07:00
  timestamps.
- `tests/unit/test_web/test_analyze_single_ticker_golden.py` golden values
  will move where the synthetic history includes AMC-era events.
- No existing test asserts the unknown→BMO bracket, so 2b contradicts nothing.

## 3. DD-2/DD-3 design — event-spanning validity for the decomposition

**Research correction (agent B): the fix site is NOT edge_engine's term
functions.** Their `near_term_*` outputs are discarded at
`edge_engine.py:2746-2758` (`_` unpack). The live selection that feeds every
displayed metric is `services/earnings_vol_snapshot.py::_build_term_structure_snapshot`
line 887 (`near = expiry_stats[0]`, expiries filtered only against
`as_of_date` — never `earnings_date`). And `decompose_event_vol` cannot
self-guard: its signature has no earnings-date input. The guard therefore
lives in `build_vol_snapshot` (`earnings_vol_snapshot.py:355-363`), which
already has `days_to_earnings` (:236) and `earnings_date` (:235).

### 3a. Event-spanning expiry (primary fix)
In `_build_term_structure_snapshot` (or immediately after it in
`build_vol_snapshot`): locate the **first expiry at/after `earnings_date`**
within the already-computed `expiry_stats` (all expiries + DTEs are in hand;
the selection mirrors `web/api/screener_engine.py::_pick_front_expiry_after`
:151-158 — reuse the comparison convention, strictly-after vs at-or-after must
match the release timing: BMO earnings on date D are spanned by expiry ≥ D;
AMC earnings on date D require expiry > D... define once, document, test).
New `VolSnapshot` fields: `event_expiry_dte`, `event_expiry_implied_move_pct`,
`event_expiry_spread_pct` (dataclass :54-140 + `to_dict` :142).

`decompose_event_vol` is then fed the EVENT-SPANNING expiry's implied move and
DTE — its math is already correct once the input spans the event. When the
near-term expiry itself spans the event (the common case), `event_expiry_* ==
near_term_*` and behavior is unchanged.

### 3b. Validity guard (backstop)
When NO quotable post-event expiry exists: `event_implied_move_pct = None`,
`non_event_move_pct = None`, `event_move_share_of_total = None`, plus
`event_decomposition_status` field: "ok" | "used_event_expiry" |
"no_event_spanning_expiry". Downstream is already largely None-tolerant
(edge_engine :2871-2945 is isfinite-guarded; `_classify_move_risk` returns
("unknown", None) on None input per edge_math.py:41-42; `MoveRiskBadge`
auto-hides — badges.jsx:95; App.jsx :942-965 null-conditioned). The scorecard
`_score_*` helpers must be checked for None-tolerance
(structure_scorecard.py:216-223) — [verified during implementation].

### 3c. Move-risk + ratio honesty
`_classify_move_risk`, `historical_vs_implied_move_ratio` (:397-398),
`tail_vs_implied_move_ratio` (:401-402), `implied_vs_anchor_ratio`
(edge_engine :2879-2884), `raw_gross_edge` (:2874-2878), and
`base_drawdown_risk` (:2902-2906) all consume the event-implied side — after
3a they automatically use the event-spanning value. The MOVE RISK banner
sub-label (App.jsx:958 / badges.jsx:102) should disclose the expiry it is
computed from ("P90/Impl 1.4x · to Jul 31").

### 3d. Near-term fields keep their meaning
`near_term_spread_pct` / `near_term_liquidity_proxy` / `near_term_dte` still
describe the FRONT tradeable contract (execution gating) — semantics
unchanged. `implied_move_pct` (Total) remains the near expiry's honest total
straddle move (research point 6): keep it, but the Full-metrics panel labels
it with its expiry and shows `Event-Expiry Implied Move` alongside when the
two differ (App.jsx :939-950 field list).

### 3e. Edge_engine dead copies
The two dead-for-metrics near-term picks (edge_engine :1713-1724, :2008-2016)
get the same spans-event awareness only if trivial; otherwise a comment
pointing at the snapshot seam ("metrics do NOT come from here") to prevent
future re-wiring from resurrecting the bug. No behavior change (outputs
discarded).

## 4. DD-4 design — regime-aware entry score (smaller, separable)

`services/screener_service.py::_iv_entry_score` currently maps trailing
iv_rv 0.80 → 1.00. Change: blend trailing and forward measures —
`iv_rv_entry_input = max(iv_rv_trailing, α·iv_rv_har)` with α = 0.75
[FINAL FORM PENDING REVIEW], or a percentile-conditioned discount: when
`rv_percentile_rank ≥ 90`, cap the entry score contribution (regime-inflated
trailing RV cannot mint a perfect score). The snapshot already carries
`iv_rv_har` and `rv_percentile_rank`; plumbing exists.
Also: when `decomp_regime_warning` is true, the ranked row should carry a
visible regime chip in the UI so #1-with-warning is legible.

## 5. DD-5 — presentation cross-reference (docs/UI only)

One page, two verdicts is by design (different strategies), but each panel
must say which strategy it speaks for. Small UI copy change: screener header
notes "pre-earnings long-vega entry ranking"; the edge-analysis No-Trade block
already names the structure. No engine change.

---

## 6. Blast radius & consumers

### DD-1 (from research agent A)
Producer chain (buggy live path A): `yfinance_market_data_client.py:255/277/282`
→ `edge_engine.py:2568/2571` (`_earnings_dates_from_mda` :1920/:1940) →
`compute_earnings_move_profile` (`earnings_move_profile.py:170-179` — only
AMC gets the shifted bracket; unknown falls into the BMO `else`).
Edge_engine also seeds its own vol snapshot with the degraded list
(`edge_engine.py:2651` `prior_events`), so on the live path the contamination
reaches everything below:

- Snapshot fields: `historical_median/avg_last4/p90/std/move_anchor_pct`
  (`earnings_vol_snapshot.py:381-386, 397-402, 523-527`), null-reason flags
  (:452-458).
- Edge metrics: implied-vs-anchor gap/ratio (:2875-2882), drawdown-risk p90
  term (:2903-2904), `_classify_move_risk` (:2941; impl `edge_math.py:30-55`,
  ratio = p90/implied — **understated p90 under-reports risk**), response
  fields (:3449-3460, :3589-3590), legacy adapter
  (`edge_legacy_adapter.py:47-51,94`).
- Screener ranking: `median_earnings_move_pct` feeds `_move_history_score`
  weight 0.25 (`screener_service.py:75,101-113,177-207,530-546`).
- Structure scorecard: anchor scoring weight 0.15 + moderate-anchor peak
  (`structure_scorecard.py:224,327,341,345,440`) — selector tiers inherit.
- Frontend: `EarningsMoveHistoryPanel` per-event bars + its own median/exceed
  computations (`earningsMoveHistory.js:33,41,54-64`), "Median Hist Move" and
  anchor narrative (`selectorViewModel.js:253,536,550`), ranked "Med Move"
  column (`RankedSetupTable.jsx:44,145`).
- NOT affected: ML feature stores/label pipeline (parallel timing derivation),
  outcome recorder (records its own timing).

### DD-2 (from research agent B)
Live producer: `earnings_vol_snapshot.py::_build_term_structure_snapshot`
(:847-953; the unchecked pick at :877/:887), `_expiry_atm_stats` (:956-1011),
decomposition invocation `build_vol_snapshot` :355-363, ratios :397-402.
Adapter: `edge_legacy_adapter.py::snapshot_to_edge_inputs` :20-64.

Consumers of the (currently fabricated, post-fix event-spanning) fields:
- Edge metrics: implied_vs_anchor (:2879-2884), raw_gross_edge (:2874-2878),
  drawdown risk (:2902-2906), `_classify_move_risk` (:2941-2945;
  edge_math.py:30-55), serialization :3446-3455/:3589-3591, rationale text
  :3277/:3318-3331/:3361-3367.
- Structure scorecard (**reaches the recommendation**): move_ratio_score
  (:216), tail_score (:217/:325/:380/:466), moderate_event_score (:223),
  rationale :279-280; `invalid_near_term_dte` flag :585-586.
- Selector thesis text: structure_selector.py:296-301.
- Ranked screener: does NOT compute event decomposition, but inherits the
  unchecked front expiry for `near_term_atm_iv`/`near_back_iv_ratio`/`iv_rv`
  (screener_service.py:525-528) — latent same-root issue, logged (see §9 Q5).
- Frontend: MoveRiskBadge (App.jsx:543; badges.jsx:94-114 — auto-hides on
  None), Full-metrics fields (App.jsx:939-965), Implied/Anchor (:767-768),
  structure P&L implied move (:652), Evidence narrative
  (selectorViewModel.js:252-254, 535-551).
- NOT affected: ML DB (uses iv30_rv30 only), alerts (none exist).

Existing correct pattern to mirror: `screener_engine.py::_pick_front_expiry_after`
(:151-158) + `_pick_next_monthly_opex` (:161-175); analogues in
scripts/forward_screener.py:114, run_forward_loop.py:694,
build_earnings_iv_labels.py:37.

Tests that pin the bug (change with the fix):
- `tests/unit/test_web/golden/analyze_watch.json` — literally near_term_dte=4 /
  days_to_earnings=8 with move_risk "elevated" AND the "not before the
  short-leg expiry" gate simultaneously — the suite asserts the category
  error. Regenerate + hand-review.
- `tests/unit/test_web/test_edge_engine_research_signals.py` — mock snapshots
  with the same pre-event shape (:104-130, :295-321, :920) + `_classify_move_risk`
  unit pins (:670-691).
- Scorecard/selector/model-card tests consuming event_implied and the ratios —
  expect assertion drift; review each rather than blind-regenerate.
- `test_event_vol_decomposition.py` — unaffected (pure math, signature
  unchanged).

## 7. Test plan

- Unit (DD-1a): yfinance client get_earnings maps historical index times →
  AMC/BMO via canonical normalizer; midnight-stamped rows stay None. Replaces
  the bug-pinning assertion at test_yfinance_market_data_client.py:137; the
  fixture gains real 16:05/07:00 timestamps. Red-then-green: assert the OLD
  behavior fails (2022-era AMC event measured on the wrong day).
- Unit (DD-1b): profile with synthetic AMC event + unknown timing → chosen
  unknown-policy behavior (bracket or drop+flag per review); BMO/AMC
  known-timing behavior byte-identical to today; `unknown_timing_event_count`
  populated. Mirror-check `_event_contaminated_session_dates` stays in
  lockstep.
- Unit (DD-2): snapshot with pre-event front expiry + post-event second
  expiry → event fields computed from the second (`event_expiry_dte` correct,
  status "used_event_expiry"); no post-event expiry → None fields + status
  "no_event_spanning_expiry"; front expiry spans event → identical to today
  (status "ok"). Spanning convention per §9 Q2 for BMO vs AMC boundary dates.
- Integration: analyze-path — golden `analyze_watch.json` REGENERATED and
  hand-reviewed (it currently asserts the DD-2 category error: move_risk
  elevated computed from a pre-event expiry while the short-leg gate fires).
  research_signals mock snapshots updated to carry the new fields.
- Convergence (DD-1): same synthetic AMC-era history through the edge adapter
  and snapshot adapter → equal medians (extend
  test_earnings_vol_snapshot.py:397-419).
- Scorecard drift: run scorecard/selector/model-card suites; review each
  numeric drift (they consume event_implied + ratios) — no blind regeneration.
- Live verify (manual): PYPL analyze returns median ≈8.6 / p90 ≈13.9, event
  decomposition reported from the 7/31 expiry (not 7/24), MOVE RISK banner
  states its expiry, history panel shows real AMC-era reactions (2022-02-01
  ≈ 25%, not 2.24%).

## 8. Rollout

Single PR, additive schema fields, no breaking response changes (existing
fields keep semantics; new status/flag fields added). Frontend changes limited
to labels/annotations + banner honesty state.

## 9. Open questions for the independent review

1. Bracket-window upward bias (2b): acceptable vs alternative (drop unknown
   events + flag)? The tradeoff: bias ~ one session of ordinary vol vs losing
   up to 100% of sample for yfinance-provider users. Research nuance: after 2a,
   "unknown" survives only for rows Yahoo stamps at midnight — so 2b is a
   backstop for a small residue, which weakens the case for the riskier
   max-of-two-windows bias and strengthens "drop + flag" (option c) since the
   dropped count should be small post-2a. DECISION DEFERRED to review.
2. Spanning convention: for BMO earnings on date D, is expiry == D
   event-spanning (report lands pre-open, so D-expiry options DO capture the
   reaction)? For AMC on date D, expiry == D does NOT span (reaction lands
   D+1). Proposed rule: spanning iff expiry_date >= reaction_session, where
   reaction_session = D for BMO/intraday/unknown-conservative?, D+1 for AMC.
   Needs review — unknown timing should probably use D+1 (conservative:
   requires strictly-after) to avoid claiming coverage that may not exist.
3. DD-4 α / cap form — max-blend vs percentile-conditioned cap.
4. Sequencing: DD-1+DD-2 in one PR (shared test scaffolding + one golden
   regeneration pass) vs two PRs (cleaner review). Leaning ONE PR because the
   golden fixture moves for both fixes and two sequential regenerations create
   confusing intermediate goldens.
5. Latent issue (logged, likely out of scope): ranked screener's
   `near_term_atm_iv`/`ts_ratio`/`iv_rv` inherit the unchecked front expiry
   (screener_service.py:525-528) — a pre-event front leg can drive entry
   scores. Fix now or as follow-up?
6. DD-1 threshold inconsistency follow-up: build_earnings_iv_labels.py uses
   AMC>=15h / BMO<=10h vs canonical >=16h / <9:30. Out of scope here; ticket.
