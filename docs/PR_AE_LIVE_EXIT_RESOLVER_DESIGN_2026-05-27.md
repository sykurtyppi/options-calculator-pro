# PR-AE — Live Candidate Exit Resolver (Design)

**Status**: Design draft — rev 2 (Codex design review applied).
**Author**: PR-AE planning session. **Date**: 2026-05-27.
**Companion governance doc**: `docs/CALENDAR_PICKER_PROMOTION_2026-05-27.md`.

### Revision history

- **rev 1** — initial draft. Proposed a lazy `from web.api.edge_engine
  import _simulate_candidate_shadow_outcome` inside the resolver
  function.
- **rev 2 (this revision)** — applies the Codex design-review
  required changes:
  1. Extract `_simulate_candidate_shadow_outcome` into a neutral
     `services/candidate_shadow_outcome.py` module **before** the
     resolver lands. Both `web/api/edge_engine.py` and the resolver
     import it. No services → web dependency, even lazily.
  2. Idempotent update semantics spelled out in full: identity key,
     duplicate-run behavior, content-hash short-circuit, attempt-
     counter rules on no-op runs.
  3. Telemetry counts taxonomy made explicit.
  4. Added historical-replay-tag-rejection regression test.

This document is the design contract for PR-AE. **It is intentionally
scoped narrowly**: PR-AE wires up live-forward provenance tagging and
adds a post-event resolver that fills in `candidate_shadow_outcome` on
existing live ledger rows. PR-AE does **not** change selector behavior,
scoring, default picker, put-calendar candidate support, or any
execution-realism (spread/fill) logic. Those are deferred to later PRs.

The single thing PR-AE unlocks is: the
`promotion_eligible_candidate_stats` block in the aggregator —
currently empty by construction on every data source — can finally
start filling with real forward observations.

---

## TL;DR

The current state, after PR-AC + PR-AD merged:

- Live forward recommendations are recorded into the ledger with
  `sample_provenance = NULL` (because `_tag_live_forward_observation`
  is imported into `web/api/edge_engine.py` but **never called**).
- Live forward recommendations carry an **empty**
  `candidate_shadow_outcome` (entry chain exists at write time; exit
  chain does not yet exist).
- `is_promotion_eligible(record)` therefore returns `False` for
  every live row today, by both the provenance check and the
  outcome-status check.

PR-AE closes both halves of that gap:

1. **At write time (live API path)**: tag the record with
   `SAMPLE_PROVENANCE_FORWARD_POST_FREEZE` via the underscore-prefixed
   `_tag_live_forward_observation` helper, preserving the
   source-grep assignment boundary.
2. **Post-event (scheduled resolver)**: a new service walks ledger rows
   where `sample_provenance == "forward_post_freeze"` AND the candidate
   outcome is still unresolved AND enough trading days have passed
   since the earnings event, and fills in the candidate shadow
   outcome by querying the post-event chain and re-running the
   existing `_simulate_candidate_shadow_outcome` simulator.

When this lands, candidate forward PnL begins accumulating. Promotion
remains gated on weeks/months of forward evidence plus put-side
parity plus execution realism (none of those are in PR-AE).

---

## Non-goals (explicit)

PR-AE MUST NOT include any of:

- ❌ Promotion of the `+14d` rule, or any change to the default picker.
- ❌ Any change to selector scoring or structure choice.
- ❌ Put-calendar candidate support
  (`put_side_not_yet_supported` placeholder stays in place).
- ❌ Spread-adjusted fills, conservative execution pricing, stale-quote
  detection, or liquidity bucketing. All outcomes remain
  `research_mid` / `not_execution_grade`.
- ❌ Any net-new ledger column except what's strictly required to
  track resolver attempts/status (single attempt counter column,
  see schema section).
- ❌ Any change to `_simulate_candidate_shadow_outcome`'s pricing
  semantics. The resolver calls it as-is.

If any of these tempt during implementation, the answer is "next PR."

---

## Background grounded in the code

Numbers below cite the merged state of `main` as of 2026-05-27.

**Tagging boundary**
- Constant: `services/candidate_shadow_provenance.py:56`
  (`SAMPLE_PROVENANCE_FORWARD_POST_FREEZE = "forward_post_freeze"`).
- Sole assigner: `_tag_live_forward_observation`
  (`services/candidate_shadow_provenance.py:208–232`).
- Source-grep regression test
  (`TestForwardProvenanceAssignmentBoundary`) scans `services/`,
  `web/`, and `scripts/` for any reference to the helper name OR the
  literal `"forward_post_freeze"`. PR-AE's new call site must live in
  `web/api/edge_engine.py` — anywhere else fails the boundary test.

**Live recommendation write path**
- Constructor: `EdgeSnapshot` at `web/api/edge_engine.py:326`.
- Build site: `web/api/edge_engine.py:4284–4294`.
- Ledger write: `web/api/edge_engine.py:4297–4302`
  (`record_recommendation(edge_snapshot, ...)`).
- `EdgeSnapshot` currently has no `sample_provenance` or
  `candidate_shadow_outcome` attribute, so
  `build_record_from_analysis` reads them as `None` / `{}`.

**Simulator the resolver will reuse, unchanged**
- `_simulate_candidate_shadow_outcome`
  (`web/api/edge_engine.py:647–779`).
  Inputs: `entry_chain` (DataFrame), `exit_chain` (DataFrame),
  `dual_picker` (Dict).
  Returns a stable-shape dict with one of 11 `status` enums.

**Reusable chain reader**
- `services/options_feature_store.OptionsFeatureStore.query_chain`
  (`services/options_feature_store.py:146–202`).
  Returns a pandas DataFrame; empty DataFrame on no rows; no
  exception on missing partition for the symbol.

**Ledger schema + revisions (PR-K) pattern**
- Append-only `recommendations` table; immutable first-write semantics.
- Update-shaped writes go to `recommendation_revisions` via
  `UNIQUE(recommendation_id, content_hash)` dedupe.
- Idempotent re-records of byte-identical content become no-ops.
  The resolver MUST use this path — it must not UPDATE the original
  `recommendations` row.

**Earnings event date**
- `recommendations.earnings_date` is a TEXT column, ISO YYYY-MM-DD,
  already indexed (`services/recommendation_ledger.py:117–118`).
- Resolver eligibility queries can be cheap range scans on
  `earnings_date`.

**Existing scheduled scripts**
- One-shot CLI shape: `scripts/watch_daily_evidence_cycle.py`.
- Forward loop: `scripts/run_forward_loop.py`.
- Logs land under `~/.options_calculator_pro/logs/`.

---

## Architecture overview

```
                       ┌──────────────────────────────────────┐
   Live API path       │ web/api/edge_engine.py               │
   (write time)        │   analyze_single_ticker(...)         │
                       │     ├── build EdgeSnapshot           │
                       │     ├── _tag_live_forward_observation│  ← NEW (PR-AE half 1)
                       │     │   on a small live_record dict  │
                       │     ├── attach .sample_provenance    │
                       │     │   and empty .candidate_shadow_ │
                       │     │   outcome to the snapshot      │
                       │     └── record_recommendation(...)   │
                       └──────────────────────────────────────┘
                                       │
                                       ▼
                              recommendation row
                              ├ sample_provenance = forward_post_freeze
                              ├ candidate_shadow_outcome = {}
                              └ earnings_date = YYYY-MM-DD


       (≥ N trading days after earnings_date)

                                       │
                                       ▼
                       ┌──────────────────────────────────────┐
   Resolver path       │ services/candidate_exit_resolver.py  │  ← NEW (PR-AE half 2)
   (post-event)        │   resolve_pending_candidate_exits()  │
                       │     ├── eligibility query            │
                       │     ├── pick post-event trade_date   │
                       │     ├── query exit chain (T9)        │
                       │     ├── rebuild dual_picker from     │
                       │     │   picker_provenance_json       │
                       │     ├── _simulate_candidate_shadow_  │
                       │     │   outcome(entry_chain,         │
                       │     │   exit_chain, dual_picker)     │
                       │     ├── update via ledger.record()   │
                       │     │   → goes to revisions table    │
                       │     └── emit JSONL telemetry row     │
                       └──────────────────────────────────────┘
                                       │
                                       ▼
                              recommendation_revisions row
                              ├ content_hash differs from v1
                              └ record_json now has candidate
                                 outcome populated
                                       │
                                       ▼
                          aggregator sees promotion-eligible
                          row via is_promotion_eligible(record)
```

The boundary the resolver respects:

- It is a **reader + revisor**, not an originator. It never sets
  `sample_provenance` — that already happened at write time.
- It never modifies the original `recommendations` row. It always
  goes through `RecommendationLedger.record()` so that updated
  payloads land in `recommendation_revisions` as version N+1.
- It never re-resolves the legacy outcome (legacy already has its own
  resolution path). It only fills in the candidate side.

---

## Eligibility (which ledger rows get resolved)

A ledger row is eligible for resolution iff ALL of these hold. Each
check is fail-closed.

1. **Origin**: `sample_provenance == "forward_post_freeze"`.
   (Historical-replay rows already carry resolved candidate outcomes
   and must not be touched.)
2. **Has a picker selection to resolve**:
   `picker_provenance_json` parses as a dict and contains a usable
   candidate selection block (`pickers_diverged` may be True or
   False — even when pickers agree, the candidate outcome is still
   computed for symmetry; the aggregator handles dedup elsewhere).
3. **Candidate outcome not yet ok**: the latest revision's
   `candidate_shadow_outcome.status` is not `"ok"` and not in the
   terminal permanently-failed set (see "Permanent failure" below).
4. **Selected structure**: `selected_structure` ∈
   {`call_calendar`}. Put-calendar deferred (non-goal).
5. **Earnings event is past, by enough margin**:
   `earnings_date` is a parseable date AND
   `today - earnings_date >= MIN_DAYS_AFTER_EVENT` calendar days
   (default 2). The margin exists so we land on a real post-event
   trade_date with quotes settled, not the morning of the announcement.
6. **Symbol availability**: the T9 feature store reports the symbol
   exists (`OptionsFeatureStore.is_available()` plus a coverage
   check). Otherwise mark as `awaiting_chain_data` (transient, retry
   eligible).
7. **Not at max attempts**: `candidate_exit_resolver_attempts < MAX_ATTEMPTS`
   (default 6).

Default constants live in `services/candidate_exit_resolver.py`
and are exported for tests to override:

```python
MIN_DAYS_AFTER_EVENT: int = 2
MAX_LOOKAHEAD_TRADE_DAYS: int = 5  # how far past earnings_date to look for a chain
MAX_ATTEMPTS: int = 6              # ≈ ~1 week of daily retries before permanent_failed
```

The eligibility query lives in `recommendation_ledger.py` as a new
public method `list_pending_candidate_exit_resolutions(...)` returning
the latest revision payload (not the immutable v1) for each candidate
recommendation_id. Callers must operate on the latest payload to see
prior resolver attempts.

---

## Post-event chain date selection

The candidate simulator wants an `exit_chain` DataFrame at the post-
event trade_date. The strategy:

1. Build `candidate_dates`: list of business-day dates from
   `earnings_date + 1 BDay` to `earnings_date + MAX_LOOKAHEAD_TRADE_DAYS BDay`,
   inclusive. Use `pandas.tseries.offsets.BDay`. (This is a pure
   weekday calendar — it skips weekends but not US market holidays.
   The lookahead horizon of 5 BDays absorbs typical holidays without
   needing an exchange calendar dependency.)

2. For each date in `candidate_dates`, call
   `store.query_chain(symbol, trade_date=date)`. Return the first
   non-empty frame found.

3. If none of the candidate dates returned rows:
   - If `today - earnings_date < MAX_LOOKAHEAD_TRADE_DAYS` calendar
     days: status is `awaiting_chain_data` (transient). Do not
     increment `candidate_exit_resolver_attempts`; the row stays
     eligible.
   - If `today - earnings_date >= MAX_LOOKAHEAD_TRADE_DAYS` calendar
     days AND attempts < `MAX_ATTEMPTS`: status is `retrying`.
     Increment the attempt counter.
   - If attempts have hit `MAX_ATTEMPTS`: status is
     `permanently_failed:no_post_event_chain`.

We deliberately do **not** try entry-chain inference, alternate
expiries, neighboring symbols, or any "best-effort" substitution.
If the post-event chain isn't there, the resolver reports it and
moves on. Substitution would distort the very evidence the resolver
exists to produce.

We also deliberately do **not** pick the latest available chain past
the window (e.g. "any chain ≤ 10 BDays post-event will do") — the
candidate simulator's IV-crush interpretation depends on outcomes
being landed at a roughly consistent horizon. Picking a 9-BDay-late
exit because the 3-BDay-late chain was missing would mix horizons in
the aggregator. Stick to the window or mark permanently failed.

---

## Rebuilding the dual_picker input for the simulator

`_simulate_candidate_shadow_outcome` expects a `dual_picker` dict with
a `candidate_selection` sub-dict carrying `side`, `front_expiry`,
`back_expiry`, and `strike`. The PR-AC `picker_provenance_json`
column on the recommendations row already carries the
`experimental_contract_selection` block, which has exactly those
fields (PR-AC commit 3 surface).

The resolver loads `picker_provenance_json` from the latest revision
payload and synthesizes a minimal dual_picker dict:

```python
dual_picker = {
    "shadow_status": "ok",
    "candidate_selection": {
        "side": picker_provenance["candidate"]["side"],
        "front_expiry": picker_provenance["candidate"]["front_expiry"],
        "back_expiry": picker_provenance["candidate"]["back_expiry"],
        "strike": picker_provenance["candidate"]["strike"],
    },
}
```

If `picker_provenance_json` is missing or malformed, the resolver
records `permanently_failed:no_picker_provenance` for that row — it
cannot recover the candidate's intended contracts without it.

The resolver does NOT need the entry-chain DataFrame to be re-derivable
from quotes alone — the picker_provenance already names the exact
contracts. But the simulator still needs both entry and exit chains
because mid-pricing for the entry leg has to come from the *real*
entry-date quotes (not the prices recorded in `bid_ask_mid_json`,
which are computed post-spread-fill rather than chain mids). So:

- **Entry chain** is re-queried at
  `store.query_chain(symbol, trade_date=record["as_of_date"])`. If
  empty (rare — usually the chain that was just used to build the
  recommendation), mark `permanently_failed:no_entry_chain_replay`.
  This case implies underlying data was retroactively pruned/changed
  and the resolver has nothing reproducible to work from.
- **Exit chain** is selected per the algorithm in the prior section.

---

## Idempotent update behavior

The resolver writes its result through `RecommendationLedger.record(
updated_record)`. Existing PR-K semantics give us idempotency mostly
for free; this section spells out the exact contract per Codex
design-review point #2.

### 1. Identity key

Each resolution targets exactly one ledger row identified by
`recommendation_id`. The resolver:

- Reads the row's **latest revision payload** via
  `ledger.get_with_latest_resolution(recommendation_id)` (the new
  helper). It must read the merged latest payload — not the
  immutable v1 — so that prior partial-resolver attempts (e.g. a
  prior `retrying` revision) are visible.
- Reconstructs a `RecommendationRecord` from that payload via the
  existing `build_record_from_analysis` path applied to a synthetic
  analysis dict — or, more directly, by mutating the payload's
  `candidate_shadow_outcome` field and passing the result to a new
  helper `RecommendationLedger.record_resolution_payload(payload)`
  that bypasses re-deriving and writes through to `record(...)`.
- The `recommendation_id` on the updated record is the same as the
  original — never regenerated. Salt is left blank. This is what
  routes the write into the revisions table rather than creating a
  new row.

The `make_recommendation_id` function takes a salt only on
*original* writes; the resolver never calls it.

### 2. How duplicate resolver runs avoid duplicate revisions

`RecommendationLedger.record()` is already PR-K-correct:

- The `recommendations` table write uses `INSERT OR IGNORE`. Since
  the row already exists, this is always a no-op for the resolver
  (rowcount=0). Original evidence stays exactly as written at
  `analyze_single_ticker` time.
- The `recommendation_revisions` table write attempts `INSERT OR
  IGNORE` against the `UNIQUE(recommendation_id, content_hash)`
  constraint. A re-run that produces a byte-identical payload
  (identical `content_hash`) is silently de-duped: no new revision
  row, `record()` returns `"duplicate"`.

Therefore: running the resolver twice in a row, with no upstream
data change between runs, produces exactly **one** revision row.

### 3. Content-hash short-circuit semantics

`_record_content_hash` (already implemented at
`services/recommendation_ledger.py:714–726`) excludes `created_at`
and `recommendation_id` from the hash. It includes everything else.
This means:

- A re-resolution with identical chain data, identical simulator
  output, and an identical resolved `candidate_shadow_outcome` →
  same content_hash → `"duplicate"` return → no new revision row.
- A re-resolution after T9 data drift (e.g., a reprocessed quote
  changes one mid) → different content_hash → `"revision"` return
  → new revision row lands. This is the correct audit-trail behavior:
  the trail should show that the resolved outcome changed because
  upstream data changed.
- A re-resolution that lands the same outcome but flips the
  resolver_status from `awaiting_chain_data` to `ok` is, by
  definition, a payload change (status string differs) → new
  revision row. So the transition `awaiting → ok` always leaves a
  visible audit row, even if no PnL fields changed.

`record_resolution_payload` returns the three-state `RecordStatus`
from `record()`: `"inserted"` (only possible on a previously-missing
recommendation_id — defensive; should never happen in resolver
context and is logged at WARNING if it does), `"revision"`, or
`"duplicate"`.

### 4. Attempt-counter rules on no-op runs

`candidate_exit_resolver_attempts` is incremented by exactly **one**
when, and only when, the resolver writes a `"revision"` whose
resolver_status is a terminal-failure status (`retrying` or
`permanently_failed:*`). The full rule table:

| Run outcome | New revision? | Attempt counter? |
|---|---|---|
| Eligibility filter excluded the row (already `ok` or terminal) | no | unchanged |
| Resolver computed `ok` (first time) | yes | unchanged (success is not a retry) |
| Resolver computed `ok` (already `ok`, but content_hash matched → duplicate) | no | unchanged |
| Resolver computed `awaiting_chain_data` (window not elapsed) | yes (if state changed) | unchanged |
| Resolver computed `retrying` (window elapsed, chain still missing) | yes | **+1** |
| Resolver computed `permanently_failed:*` | yes | **+1**, then row becomes ineligible |

The two key non-obvious rules:

- **`ok` does not increment.** A successful resolution is not a
  retry. The counter exists to bound failure scenarios, not to count
  total runs.
- **`awaiting_chain_data` does not increment.** This is "data isn't
  available yet" — burning retry budget on it would cause the
  resolver to give up before the data shows up. Already noted in
  the eligibility section; restated here so both sections agree.

The counter is incremented via the new
`RecommendationLedger.increment_resolver_attempts(recommendation_id)`
helper, in the same transaction as the `record()` write. The
counter UPDATE is the **single** mutation path PR-AE adds to the
immutable `recommendations` row — justified because the counter is
process bookkeeping rather than evidence (see "Schema additions"
section).

### 5. Concurrent run safety

The existing module-level `_WRITE_LOCK` (single in-process mutex)
plus `PRAGMA busy_timeout=5000` (cross-process backpressure on the
SQLite file) cover the daily-tick cadence. The transaction boundary
in `_tx` ensures: either both the revision insert AND the counter
increment land, or neither does. Two concurrent resolver invocations
on the same recommendation_id are safe — the second one sees the
first's revision (different content_hash means a new revision, same
means duplicate) and either way the counter increments by the
intended amount, not double.

### 6. The "current" view of a recommendation

A consumer that wants "the immutable evidence at recommendation
time" calls `ledger.get(recommendation_id)` — unchanged.

A consumer that wants "the latest known state including any
resolver-filled outcome" calls
`ledger.get_with_latest_resolution(recommendation_id)`, which:

- Loads v1 from `recommendations`.
- Loads the latest row from `recommendation_revisions` for that id.
- Returns v1 verbatim, with `candidate_shadow_outcome` overridden by
  the latest revision's payload field. Every other field of v1
  remains exactly v1 — including `sample_provenance` (which is
  immutable post-tagging) and `picker_provenance_json` (immutable
  selection record).

The aggregator's `is_promotion_eligible(record)` check operates on
this merged view. The aggregator's
`promotion_eligible_candidate_stats` block likewise consumes the
merged view.

---

## Resolver-status taxonomy

The resolver writes one of these into
`candidate_shadow_outcome.status` (orthogonal to the simulator's own
status enum):

| Status | Semantics | Eligible to retry? |
|---|---|---|
| `ok` | simulator returned `ok`, candidate PnL resolved | no — terminal success |
| `awaiting_chain_data` | post-event window not fully elapsed yet | yes, on next tick |
| `retrying` | post-event window elapsed, no chain found, attempts < MAX | yes |
| `permanently_failed:no_post_event_chain` | exhausted attempts | no |
| `permanently_failed:no_entry_chain_replay` | entry chain unrecoverable | no |
| `permanently_failed:no_picker_provenance` | candidate selection unrecoverable | no |
| `permanently_failed:simulator_error` | simulator raised an exception | no |
| `permanently_failed:simulator_<sim_status>` | simulator returned a terminal-skipped status (negative debit, malformed selection, etc.) | no |

`is_promotion_eligible(record)` already requires
`candidate_shadow_outcome.status == "ok"` (per
`services/candidate_shadow_provenance.py:192–193`). All non-`ok`
resolver statuses correctly fail that check — no aggregator changes
required.

The `awaiting_chain_data` status does **not** increment
`candidate_exit_resolver_attempts`. Only true post-window failures
do. This prevents "data not yet available" cases from burning
through the retry budget before the data shows up.

The `retrying` status is recorded so dashboards can see how many rows
are mid-retry. After `MAX_ATTEMPTS` retries with no success it
transitions to `permanently_failed:no_post_event_chain`.

---

## Schema additions

One new ledger column is strictly necessary:

```sql
ALTER TABLE recommendations
ADD COLUMN candidate_exit_resolver_attempts INTEGER NOT NULL DEFAULT 0;
```

Goes into both `_TABLE_DDL` and `_MIGRATION_COLUMNS` for backward-
compatible migration on existing databases.

This counter must live on the immutable `recommendations` row (not
inside the JSON of a revision) because the resolver must read it
atomically when deciding eligibility and increment it on each
post-window attempt. Storing it on revisions would force the resolver
to scan revision history, and concurrent resolvers could miss each
other's increments.

This is the **only** column update path that mutates the original
`recommendations` row in PR-AE. It is justified because:

- The counter is a process bookkeeping field, not part of the
  recommendation evidence. It cannot rewrite history because the
  resolver only ever increments it; older counter values aren't
  semantically meaningful.
- Treating it as evidence (and routing it through revisions) would
  conflate "this row was retried again" with "the candidate outcome
  was updated."

A short `UPDATE` statement guarded by the
`recommendations.recommendation_id = ?` WHERE clause covers the
write. Wrapped in the existing `_WRITE_LOCK` for in-process
serialization plus `PRAGMA busy_timeout=5000` for cross-process.

No new index. The eligibility query is cheap enough on
`earnings_date` + `sample_provenance` (we add a composite index
covering both).

```sql
CREATE INDEX IF NOT EXISTS idx_recommendations_provenance_earnings
    ON recommendations (sample_provenance, earnings_date);
```

---

## Module boundaries

PR-AD commit 4 just finished removing the services → web dependency
by extracting provenance constants into
`services/candidate_shadow_provenance.py`. PR-AE applies the same
pattern to the candidate shadow outcome simulator. There is no lazy
import, no runtime-only edge, no future-PR cleanup deferred — the
extraction happens up front as commit 2 of this PR.

```
services/candidate_shadow_outcome.py        ← NEW (extracted from
                                              edge_engine.py)
    imports:
      - services.candidate_shadow_provenance: (_CANDIDATE_SHADOW_LABELS,
        _CANDIDATE_SHADOW_OUTCOME_FIELDS, _empty_candidate_shadow_outcome)
      - numpy, pandas
    exports:
      - simulate_candidate_shadow_outcome(*, entry_chain, exit_chain,
        dual_picker)  ← byte-for-byte the current
        _simulate_candidate_shadow_outcome from edge_engine.py,
        renamed without the leading underscore because it is now a
        public service-layer function.
      - _lookup_exact_contract_row, _safe_float (the private helpers
        the simulator needs — moved alongside.)

services/candidate_exit_resolver.py          ← NEW
    imports:
      - services.candidate_shadow_provenance: (is_promotion_eligible,
        SAMPLE_PROVENANCE_FORWARD_POST_FREEZE, _empty_candidate_
        shadow_outcome)  -- read-only checks + factory
      - services.candidate_shadow_outcome: simulate_candidate_
        shadow_outcome   -- the simulator, neutrally located
      - services.options_feature_store: OptionsFeatureStore
      - services.recommendation_ledger: RecommendationLedger,
        RecommendationRecord
    NO imports from web/. The resolver's complete import graph
    stays inside services/ + stdlib + numpy/pandas/duckdb.

services/recommendation_ledger.py            ← EXTENDED
    + candidate_exit_resolver_attempts column (DDL + migration)
    + list_pending_candidate_exit_resolutions(...) public method
    + get_with_latest_resolution(recommendation_id) helper
    + increment_resolver_attempts(recommendation_id) helper

web/api/edge_engine.py                       ← EXTENDED + REFACTORED
    + _simulate_candidate_shadow_outcome is removed; replaced by a
      one-line `from services.candidate_shadow_outcome import
      simulate_candidate_shadow_outcome as _simulate_candidate_
      shadow_outcome` alias to keep historical-replay call sites
      identical (zero diff at simulator-call lines).
    + analyze_single_ticker now calls _tag_live_forward_observation
      on a small live_record dict, then attaches both
      sample_provenance and an empty candidate_shadow_outcome to
      the EdgeSnapshot before record_recommendation.
    + EdgeSnapshot gains sample_provenance: Optional[str] and
      candidate_shadow_outcome: Dict[str, Any] attributes.

scripts/resolve_candidate_exits.py           ← NEW
    one-shot CLI shaped like watch_daily_evidence_cycle.py;
    iterates eligible rows, calls the resolver, emits JSONL log.
```

The extraction is byte-equivalent: identical function body, identical
inputs and return shape. All existing PR-AD tests
(`_simulate_candidate_shadow_outcome`'s behavioral suite) continue
to pass by changing only the import line at the test site. No
selector / scoring / default-picker code paths see a behavior change.

### Why the extraction is non-negotiable here

Codex's design-review point on this is correct and load-bearing.
A lazy `from web.api ... import ...` inside a `services/` module is
still a services → web edge — it's just a runtime one that hides
from static analysis. The PR-AD commit 4 architectural invariant
("services has no dependency on web/api") is enforced by reading
the import graph; a lazy import would silently break that invariant
without flagging in any obvious way. The `services/` layer must be
self-contained for the same reason ledger writes go through the
neutral provenance module rather than back-importing constants from
edge_engine.

### Public name change

The extracted function loses the leading underscore:
`simulate_candidate_shadow_outcome` (public, callable from any
services-layer module). Inside `edge_engine.py` it's re-aliased to
`_simulate_candidate_shadow_outcome` so historical-replay call sites
inside that file keep their existing private-name convention.
External callers (the resolver) use the public name.

---

## Telemetry and logging

Per-row resolution outcomes write to a structured JSONL file:

```
~/.options_calculator_pro/logs/candidate_exit_resolutions.jsonl
```

One line per row processed. Schema:

```json
{
  "ts": "2026-05-29T13:45:01Z",
  "recommendation_id": "rec_abc123...",
  "symbol": "AAPL",
  "earnings_date": "2026-05-27",
  "exit_trade_date": "2026-05-29",      // null if no chain found
  "attempt_number": 2,
  "resolver_status": "ok",               // status from taxonomy above
  "simulator_status": "ok",              // from _simulate_candidate_shadow_outcome
  "mid_realized_return_pct": 12.4,       // null if status != ok
  "duration_ms": 84
}
```

JSONL is chosen so daily roll-ups can be done with
`jq -s 'group_by(.resolver_status)'` and the existing watchdog
script pattern can post-process it.

A run summary writes to stdout for the CLI invocation, suitable for
launchd/cron capture. The count taxonomy is fixed and the sum of
non-`scanned` counts must equal `scanned` minus `skipped_*`:

```
PR-AE resolver run @ 2026-05-29T13:45:01Z
  scanned:                 42 candidate rows from ledger
  skipped_ineligible:       2  (terminal status / wrong structure / etc.)
  skipped_malformed:        1  (picker_provenance missing/malformed)
  resolved_ok:             18
  awaiting_chain_data:     12
  retrying:                 5
  permanently_failed:       4  (broken out below)
    no_post_event_chain:    2
    no_entry_chain_replay:  1
    simulator_error:        1
  duration:                3.2s
```

Invariants the CLI asserts before printing the summary (and exits
non-zero if violated, so a launchd/cron tail-watcher catches drift):

```
scanned
  == skipped_ineligible
   + skipped_malformed
   + resolved_ok
   + awaiting_chain_data
   + retrying
   + permanently_failed_total
```

These counts are aggregable into the existing daily evidence cycle
watchdog if desired in a future PR.

Errors during resolution (unexpected exceptions from the simulator,
DuckDB failures, etc.) are caught in the per-row try/except so one
bad row does not abort the run. The exception is logged at WARNING
level via `utils.logger.setup_logger` and the row is recorded with
`resolver_status = "permanently_failed:simulator_error"` plus an
`error_message` field in the JSONL telemetry only (not persisted
back into the ledger payload).

---

## Tagging precursor (PR-AE half 1)

Two source edits inside `web/api/edge_engine.py:analyze_single_ticker`:

1. **EdgeSnapshot dataclass** (line 326): add fields

   ```python
   sample_provenance: Optional[str] = None
   candidate_shadow_outcome: Dict[str, Any] = field(default_factory=dict)
   ```

2. **Call site** (around line 4283): before `record_recommendation`,
   build a small dict and pass it through the underscore-prefixed
   helper. This preserves the source-grep boundary test — the
   reference to `_tag_live_forward_observation` lives in
   `edge_engine.py`, which is on the boundary's allow-list.

   ```python
   # PR-AE: tag live forward observation before ledger write. The
   # underscore-prefixed helper is the SINGLE authorized assigner of
   # SAMPLE_PROVENANCE_FORWARD_POST_FREEZE. See
   # services/candidate_shadow_provenance.py docstring.
   live_record: Dict[str, Any] = {}
   _tag_live_forward_observation(live_record)
   edge_snapshot.sample_provenance = live_record["sample_provenance"]
   edge_snapshot.candidate_shadow_outcome = _empty_candidate_shadow_outcome(
       "awaiting_exit_resolver"
   )
   ```

`build_record_from_analysis` already reads
`analysis.sample_provenance` and `analysis.candidate_shadow_outcome`
via `_get(...)`, so no ledger-side changes are required for the live
write path beyond the new schema column.

The pre-resolution status `"awaiting_exit_resolver"` is a NEW
status string the resolver recognizes as "this row is mine to
resolve." Added to the documented set of allowed status values in
`services/candidate_shadow_provenance.py` (a one-line comment update
plus the existing `_empty_candidate_shadow_outcome` factory call —
no code change to the factory itself).

---

## Test plan

All tests under `tests/unit/test_services/test_candidate_exit_resolver.py`
unless noted. Target ~25 tests.

**Eligibility filter** (8 tests)
- Excludes `sample_provenance == HISTORICAL_REPLAY`.
- Excludes `sample_provenance == None`.
- Excludes `sample_provenance == UNKNOWN`.
- Excludes rows whose latest revision already has `status == "ok"`.
- Excludes rows whose latest revision is `permanently_failed:*`.
- Excludes rows with `today - earnings_date < MIN_DAYS_AFTER_EVENT`.
- Excludes rows with `selected_structure != "call_calendar"`.
- Excludes rows where `picker_provenance_json` is empty/malformed.

**Chain date selection** (5 tests)
- Picks the first BDay with a non-empty chain frame.
- Skips empty BDays correctly through the lookahead window.
- Returns `awaiting_chain_data` when window not yet elapsed and no
  chain found (no attempt-counter increment).
- Returns `retrying` when window elapsed and attempts < MAX
  (attempt-counter incremented by exactly 1).
- Returns `permanently_failed:no_post_event_chain` at attempt = MAX.

**Idempotency** (4 tests)
- First successful resolution writes a revision; v1 row unchanged.
- Re-running on a row with `status == "ok"` is a no-op (no new
  revision, no counter change).
- Re-running with simulated upstream data drift writes a new revision.
- Running with an explicitly stubbed simulator that always returns
  `ok` and re-running with the same stub writes exactly one
  revision (idempotent via content_hash).

**Status transitions / simulator integration** (4 tests)
- Simulator `ok` → resolver records `ok` and `mid_realized_return_pct`
  flows through unchanged.
- Simulator `skipped:negative_debit` → resolver writes
  `permanently_failed:simulator_skipped:negative_debit`.
- Simulator raises an exception → resolver writes
  `permanently_failed:simulator_error`, exception is logged at
  WARNING, run continues with next row.
- Resolved record with `status == "ok"` flows through
  `is_promotion_eligible` as True (end-to-end aggregator
  integration).

**Tagging boundary regression** (3 tests, in
`tests/unit/test_web/test_calendar_dual_picker.py`)
- Update `TestForwardProvenanceAssignmentBoundary` to expect the new
  call site in `web/api/edge_engine.py` and no other new locations.
  In particular: the resolver module
  (`services/candidate_exit_resolver.py`) must contain **zero**
  references to `_tag_live_forward_observation` and **zero**
  references to the literal `"forward_post_freeze"` (the resolver
  only reads pre-tagged rows).
- New test: `analyze_single_ticker` invoked via mock returns an
  `EdgeSnapshot` whose `sample_provenance ==
  SAMPLE_PROVENANCE_FORWARD_POST_FREEZE`.
- **New (Codex watch item)**: historical-replay rejection test.
  Run `_simulate_pre_earnings_calendar_trade` end-to-end with stub
  chains and assert the produced record's `sample_provenance ==
  SAMPLE_PROVENANCE_HISTORICAL_REPLAY` — never `forward_post_freeze`,
  regardless of when the function is executed. Pairs with the
  existing source-grep test to give both static and behavioral
  coverage that historical paths cannot acquire the forward tag.

**Schema migration** (2 tests, in
`tests/unit/test_services/test_recommendation_ledger.py`)
- Opening a pre-PR-AE sqlite file (synthetic — created without the
  new column) auto-migrates to add `candidate_exit_resolver_attempts`
  defaulted to 0.
- Existing rows in the pre-PR-AE file get default `0` for the new
  column on read.

Total expected new test count: ~25, taking the suite from 619 to ~644
unit tests.

---

## Anti-patterns / what NOT to do

- **Do not** add a "force resolution" CLI flag that lets an operator
  manually tag a row's outcome. The resolver's only authority is to
  compute outcomes from chain data; manual overrides defeat the
  audit trail.

- **Do not** widen `PROMOTION_ELIGIBLE_PROVENANCES`. PR-AE produces
  the evidence; promotion criteria stay frozen at "FORWARD_POST_FREEZE
  only" until governance reviews accumulated forward evidence
  separately.

- **Do not** add a "best effort" fallback that pulls the latest
  available chain past `MAX_LOOKAHEAD_TRADE_DAYS` if the in-window
  ones are missing. That distorts the IV-crush interpretation by
  silently mixing horizons.

- **Do not** treat `is_promotion_eligible(record)` returning True
  for a single forward record as a promotion signal. The promotion
  criteria in `docs/CALENDAR_PICKER_PROMOTION_2026-05-27.md` require
  a sample-size threshold that does not yet exist; resolver landing
  is necessary but nowhere near sufficient.

- **Do not** parallelize the resolver across processes without first
  proving the SQLite writes serialize correctly. Today's daily-tick
  cadence makes one-shot single-threaded resolution sufficient.

- **Do not** call `_tag_live_forward_observation` from anywhere in
  the resolver. The resolver reads pre-tagged rows; it never assigns
  provenance. The boundary regression test enforces this.

---

## Commit sequence (proposed)

PR-AE breaks down into 6 commits. Each is independently reviewable;
the suite must pass at every commit. Commits 1–2 contain no behavior
change (schema migration plus a pure refactor); commits 3–6 land the
new behavior on top of that foundation.

**Commit 1 — Schema + ledger helpers**
- New column `candidate_exit_resolver_attempts` + migration.
- New composite index on (sample_provenance, earnings_date).
- `list_pending_candidate_exit_resolutions(...)` method.
- `get_with_latest_resolution(recommendation_id)` helper.
- `increment_resolver_attempts(recommendation_id)` helper.
- `record_resolution_payload(payload)` helper.
- Migration tests in `test_recommendation_ledger.py`.

**Commit 2 — Extract candidate-shadow-outcome simulator (refactor)**
- Create `services/candidate_shadow_outcome.py` with the byte-
  equivalent body of `_simulate_candidate_shadow_outcome`,
  renamed `simulate_candidate_shadow_outcome` (no leading
  underscore — now a public service-layer function).
- Move `_lookup_exact_contract_row` and `_safe_float` alongside it
  (or leave duplicated stubs in `edge_engine.py` if they are still
  used by other code paths there — surveyor confirms they are not
  outside the simulator).
- In `web/api/edge_engine.py`, replace the function body with a
  one-line import alias:
  `from services.candidate_shadow_outcome import simulate_candidate_shadow_outcome as _simulate_candidate_shadow_outcome`.
- No behavior change. Existing simulator tests pass unchanged.
- Add tests in `tests/unit/test_services/test_candidate_shadow_outcome.py`
  to lock in the public-name surface and reuse the existing
  behavioral test fixtures.

**Commit 3 — Live forward tagging**
- `EdgeSnapshot` fields added (`sample_provenance`,
  `candidate_shadow_outcome`).
- `_tag_live_forward_observation` call site wired into
  `analyze_single_ticker`.
- `_empty_candidate_shadow_outcome("awaiting_exit_resolver")`
  attached.
- Tagging boundary regression test updated.
- Historical-replay tag-rejection behavior test added.
- `analyze_single_ticker` returns tagged snapshot (test).

**Commit 4 — Resolver service module**
- `services/candidate_exit_resolver.py` with
  `resolve_pending_candidate_exits(ledger, store, *, now=None)`.
- Imports `simulate_candidate_shadow_outcome` from
  `services/candidate_shadow_outcome.py` — clean services-only
  import graph, no web edge.
- All eligibility / chain-date / idempotency tests.

**Commit 5 — Scheduled entrypoint + telemetry**
- `scripts/resolve_candidate_exits.py` CLI.
- JSONL telemetry writer.
- Stdout summary block with the count taxonomy from the
  Telemetry section.
- CLI asserts the count-balance invariant before exit.
- One smoke test that runs the CLI against an in-memory ledger with
  a stubbed feature store.

**Commit 6 — Aggregator end-to-end integration test + doc update**
- New end-to-end test: write a forward live record, run resolver
  against a stub chain, confirm row flows through
  `is_promotion_eligible` and lands in
  `promotion_eligible_candidate_stats`.
- Update `docs/CALENDAR_PICKER_PROMOTION_2026-05-27.md` to flip
  the "live forward candidate PnL" prerequisite from `⚠️ PENDING`
  to `✅ INFRASTRUCTURE LANDED — accumulating evidence`. Promotion
  decision criteria themselves do NOT change.

---

## Open questions deferred (not in PR-AE)

These are real questions, but they're either resolvable later or
intentionally not the resolver's problem:

- **Exchange holiday calendar**: BDay approximates well enough; if
  systematic mis-selections appear post-launch, swap to
  `pandas_market_calendars` in a follow-up.
- **Put-calendar parity**: deferred. Eligibility filter already
  excludes put_calendar rows; they accumulate as ineligible until a
  separate PR turns put_side_not_yet_supported into real support.
- **Cross-process concurrency**: single-process daily tick is
  sufficient. If a multi-worker pattern emerges later,
  `_WRITE_LOCK` and `busy_timeout=5000` already cover it for SQLite;
  worker leasing of recommendation_ids is the right pattern then.
- **Chain-data freshness**: the resolver doesn't validate that the
  exit chain reflects fully-settled quotes vs intraday snapshots. T9
  parquet is EOD per file naming, so this is implicitly handled, but
  if intraday chains start landing in the same store a freshness
  filter belongs in `OptionsFeatureStore.query_chain`, not the
  resolver.
- **Promotion governance**: explicitly stays in the existing doc;
  PR-AE doesn't move that needle.
