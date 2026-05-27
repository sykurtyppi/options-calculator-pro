# Calendar Picker Promotion Criteria — PR-AC + PR-AD

**Status**: Provisional. **Last reviewed**: 2026-05-27 (PR-AD update).

This document defines the governance rules for moving the
`candidate_min_dte` calendar leg-picker rule from experimental shadow
mode to production default. It is intentionally **conservative** about
what is measurable today versus what would require additional
infrastructure work before being evaluable.

---

## TL;DR (updated for PR-AD)

The state of the validation pipeline as of PR-AD:

- ✅ **Selection-quality** (divergence rate, coverage, stability) is
  measurable today.
- ✅ **In-sample candidate PnL** is now resolvable on the historical
  replay path (PR-AD commit 1). It is recorded into the ledger
  alongside picker selection (PR-AD commit 3). The aggregator
  surfaces it as **diagnostic only** (`in_sample_diagnostic_candidate_
  stats`) — explicitly labeled as not promotion evidence.
- ⚠️ **Live forward candidate PnL** is NOT yet resolvable. PR-AD's
  candidate shadow outcome simulator runs only inside the historical
  backtest path (`_simulate_pre_earnings_calendar_trade`). When the
  live API records an upcoming-earnings recommendation, no exit chain
  exists yet — the candidate outcome can be RESOLVED only after the
  earnings event closes, which requires a separate **exit resolver**
  service that does not yet exist.
- ❌ **Outcome-based promotion criteria** (`candidate_mean_return ≥
  legacy + 3pp`, etc.) therefore remain **not actionable**. The
  aggregator computes them but the `promotion_eligible_candidate_
  stats` block is empty by construction on every history-only data
  source. It will populate only after a live exit resolver lands AND
  enough forward earnings events accumulate.

A full promotion decision requires both the exit-resolver
infrastructure AND post-PR-AD forward observations on it. Until the
former is built, promotion-by-fiat is not allowed.

---

## Background

The `candidate_min_dte` rule (`CANDIDATE_FRONT_MIN_DTE_DAYS = 14`) was
discovered in PR-AB by sweeping holding periods across 10 symbols × 139
earnings events × 3 horizons × 4 structures and selecting the
configuration with the best in-sample mean return. On that dataset,
requiring a ≥14-day front DTE improved `put_calendar` mean return from
+12% to +23% with win rate 61% → 73%.

PR-AC introduces a dual-path shadow logger so the same rule can be
evaluated on **future** events without ever changing trading behavior.
The empirical claim "the rule lifts returns by ~11pp" is in-sample and
not validated; the system must show evidence on out-of-sample events
before the candidate becomes default.

---

## What is measurable today

After PR-AC commit 4, every recorded recommendation carries a
`picker_provenance_json` block with:

- `picker_variant` for both legacy and candidate selections
- `front_expiry`, `back_expiry`, `strike` for each
- `pickers_diverged` boolean
- `candidate_min_front_dte_days` threshold used
- Quote details (mid/bid/ask/iv/spread/OI/volume) for each leg

Aggregated across observations, this supports the following measures:

### Selection-quality measures (measurable today)

| Measure | Definition | Stable threshold for promotion-eligibility |
|---|---|---|
| `divergence_rate` | `pickers_diverged_count / picker_evaluated_count` (event-level) | **≥ 30%** — if pickers agree on 70%+ of events, the rule is essentially a no-op and divergent comparisons would be sample-thin. |
| `candidate_coverage` | `candidate_succeeded_count / legacy_succeeded_count` (event-level) | **≥ 80%** — the candidate must successfully pick contracts in at least 80% of events where legacy also picked. Below 80% means the +14d DTE floor disqualifies too many real-world setups. |
| `selection_stability` | Same event, same chain → same selection across re-runs | **100%** — the picker must be deterministic. Any randomness invalidates outcome comparisons. |

These are **pre-promotion gates**. Passing them does not prove the rule
makes money. Failing them rules the rule out as too unstable / too
narrow / too redundant to be worth promoting.

---

## What is NOT measurable today

The following criteria **cannot** be evaluated with current data
collection. Writing them as actionable thresholds would create a
governance hole exactly the size of the research-leakage problem
PR-AC was designed to prevent.

### Outcome-quality criteria (NOT measurable until prerequisites land)

- **Candidate mean return** — requires simulating the candidate picker's
  selections through entry + exit pricing on each historical event. The
  PR-AB sweep did this in-sample only; the live ledger does not.
- **Candidate win rate** — same prerequisite as mean return.
- **Per-quartile (or any conditional) candidate-vs-legacy comparison**
  — same prerequisite.

### Why not just use the PR-AB sweep numbers?

Because the PR-AB sweep was the data **on which the rule was
discovered**. Reusing it to "validate" the rule is the textbook
multiple-testing failure mode (Harvey & Liu, "Evaluating Trading
Strategies"). Out-of-sample evidence means evidence collected on events
the rule did not see during development — i.e., events after the PR-AC
merge date.

---

## Prerequisites for evaluating outcome-based criteria

Status as of PR-AD merge:

1. ✅ **Candidate shadow simulation through historical events.**
   Done in PR-AD commit 1: `_simulate_candidate_shadow_outcome` resolves
   the candidate picker's contracts through entry + exit chains and
   computes `mid_pnl` + `mid_realized_return_pct` for every historical
   simulated trade. Outputs are tagged
   `sample_provenance: historical_replay_in_sample_or_research` and are
   NEVER promotion-eligible by themselves.

2. ✅ **Schema extension.** Done in PR-AD commit 3: the recommendation
   ledger gained `candidate_shadow_outcome_json` (the full resolved
   outcome) and `sample_provenance` (denormalized for SQL filtering)
   columns. Both come with a backward-compatible migration.

3. ⚠️ **Live forward exit resolver — STILL PENDING.** This is the
   largest remaining gap. The live API records an upcoming-earnings
   recommendation at trade-entry time, but the candidate exit price
   is unknown until after the event. A separate **exit resolver
   service** must:
   - Periodically scan the ledger for forward observations whose
     event_date has passed but whose `candidate_shadow_outcome` is
     still empty.
   - Pull the post-event chain (from the existing feature store).
   - Look up the candidate's recorded contracts in that exit chain.
   - Compute `mid_pnl` + `mid_realized_return_pct`.
   - Update the ledger row with the resolved outcome.
   Until this exists, the `promotion_eligible_candidate_stats`
   block in the aggregator is empty by construction on every
   data source. Without forward outcome resolution there is NO
   way to produce promotion-grade candidate PnL — only the in-sample
   diagnostic stats exist.

4. ⚠️ **Forward-data accumulation period — gated on (3).** Once the
   exit resolver lands, observations need time to accumulate on events
   **after** the resolver's merge date. PR-AB / PR-AC / PR-AD events
   do not count — they're either in-sample (PR-AB) or recorded before
   the exit resolver existed (PR-AC/PR-AD). The clock for forward
   accumulation starts at the exit-resolver merge.

5. ⚠️ **Put-side parity — STILL PENDING.** PR-AC commit 2 only queries
   the calls chain in the historical backtest. PR-AD commit 1 inherits
   that limitation. `put_calendar` candidate evidence is therefore
   absent from the dual-path logger today. Promoting `candidate_min_dte`
   for put_calendar requires put-chain querying in the simulator AND
   in any exit resolver, plus its own forward accumulation period.
   (PR-AD commit 3 future-proofed the aggregator's event-dedupe key to
   include `option_type` so call and put outcomes for the same event
   won't collide once put-side support lands.)

Only after (3) AND (4) AND optionally (5) are done can the
outcome-based criteria below be evaluated honestly.

---

## Proposed outcome-based criteria (NOT YET EVALUABLE)

These are documented for future use. They are **not actionable today**
because the data they reference does not yet exist in the ledger.

| Criterion | Threshold | Rationale |
|---|---|---|
| Forward sample size (call_calendar) | `unique_events_picker_evaluated ≥ 40` | Minimum sample for non-trivial outcome stats. Per Harvey/Liu and SR 11-7 model-validation guidance, ≥40 distinct out-of-sample events is the smallest defensible threshold for a one-rule comparison. |
| Forward sample size (put_calendar) | Same threshold, evaluated separately after put-side parity lands | Calls and puts are different distributions — combining them is statistical sleight of hand. |
| Mean-return improvement | `candidate_mean_return_pct - legacy_mean_return_pct ≥ 3pp` | Smaller than the PR-AB in-sample lift (+11pp). Sets a deliberately lower out-of-sample bar so we don't require exact replication of the in-sample magnitude. |
| Win rate | `candidate_win_rate ≥ legacy_win_rate` | Direction matters more than magnitude. A non-decrease is the lowest defensible bar. |
| Rolling-window check | Both conditions above must hold on the most recent 20-event rolling window in addition to the cumulative sample | Guards against the rule working only in one specific market regime. |

All four conditions must be satisfied for promotion. Any single
condition failing → no promotion. A passing event count alone (e.g.,
"n ≥ 40, no other condition checked") is **not** sufficient.

---

## Promotion mechanics

When (and only when) all four outcome-based conditions above are met:

1. The reviewer changes the default `picker_variant` used by the
   public-facing recommendation surface from `legacy_first_expiry` to
   `candidate_min_dte`. **`legacy_first_expiry` is not removed** — it
   remains available so the shadow comparison can continue running
   in the *other* direction.
2. A new shadow comparison begins: the (newly-promoted) candidate
   becomes the canonical baseline, and any future picker variant must
   beat it on the same out-of-sample protocol.
3. The promotion event itself is recorded in a new `picker_promotion`
   row in the ledger, including the cumulative + rolling-window stats
   at the moment of promotion. This is the audit record.

---

## Revert mechanics

After promotion, if the post-promotion rolling-window check ever fails
(candidate underperforms legacy on the most recent 20 events), the
default is **reverted** to `legacy_first_expiry`. The revert is
automatic, not discretionary. The revert event is also recorded in the
ledger.

This is the SR 11-7 ongoing-monitoring principle: a model is not
"validated and done"; validation continues for the life of the model.

---

## Anti-patterns

The following actions are **explicitly forbidden** under this
governance:

- **In-sample re-validation.** Using the PR-AB 139-event sweep, or
  any subset of it, to argue the candidate is ready for promotion.
  That data was the discovery set; it is not a validation set.
- **Sample stitching.** Combining the in-sample PR-AB events with
  post-merge forward events into a single n-count. The thresholds in
  this document refer exclusively to events whose `created_at`
  timestamp is **after the PR-AC merge commit**.
- **Threshold drift.** Lowering the mean-return improvement threshold
  below 3pp, or lowering the sample-size threshold below 40, without
  publishing a new version of this document explaining why and
  acknowledging the multiple-testing implications.
- **Outcome reading from selection alone.** Treating a high
  `divergence_rate` or `candidate_coverage` as outcome evidence.
  These are pre-promotion gates, not performance proxies.

---

## Open governance questions

These are deliberately left unresolved in this revision of the
document. They should be settled before the prerequisites work
begins:

- Should `put_calendar` promotion require its own parity-threshold
  satisfaction independent of `call_calendar`, or should they be
  bundled? (Current draft assumes independent.)
- Should the rolling-window check be 20 events, 30 events, or 50?
  Smaller windows are noisier; larger windows are slower to react.
- Should there be a minimum "elapsed calendar time" requirement
  (e.g., ≥6 months of forward data) in addition to the event count?

---

## Document history

- 2026-05-27: Initial version (PR-AC commit 5). Conservative scope per
  Codex review of PR-AC commit 4 — outcome-based criteria documented
  but explicitly marked NOT YET EVALUABLE.
- 2026-05-27: PR-AD update. Per Codex review of PR-AD commit 2 ("do
  not overclaim that live forward validation is complete"):
  prerequisite (1) marked DONE for historical replay; prerequisite (2)
  marked DONE for ledger persistence; new prerequisite (3) **Live
  forward exit resolver — STILL PENDING** explicitly added as the
  largest remaining gap. The `promotion_eligible_candidate_stats`
  aggregator block exists but is empty by construction until forward
  exit resolution exists.
