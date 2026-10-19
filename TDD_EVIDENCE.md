# TDD Evidence

## RED

- `pytest -q tests/unit/test_services/test_evidence_funnel.py::test_same_symbol_different_event_dates_keep_distinct_discovery_rows`
  - Failed: only `AAPL|2026-05-22` remained; the first same-symbol candidate was overwritten.
- Focused new regressions initially exposed missing `reset_response_caches`, metadata-only cohort hashing, truthy string claim flags, and `OverflowError` from `datetime.fromtimestamp(1e300)`.
- `pytest -q tests/unit/test_scripts/test_alert_outbox.py::test_main_closes_outbox_on_success`
  - Failed: `AlertOutbox.close()` was never called.
- Same-as-of earnings correction regression initially had no supersession because equal `as_of_date` keys were treated as ambiguous.

## GREEN

- Focused regressions:
  - evidence candidate identity: `1 passed`
  - strict claim normalization and oversized timestamp: `9 passed`
  - response-cache path isolation: `1 passed`
  - cohort interpretation metadata tamper checks: `2 passed`
  - same-as-of earnings correction: `1 passed`
  - deterministic outbox close: success and exception paths pass
- Combined outbox + delivery order/WAL suite:
  - `pytest -q tests/unit/test_scripts/test_alert_outbox.py tests/unit/test_scripts/test_alert_delivery.py`
  - `40 passed in 0.91s`
- Relevant grouped suites, including populated store migrations:
  - `pytest -q tests/unit/test_services/test_evidence_funnel.py tests/unit/test_services/test_claim_contract.py tests/unit/test_services/test_claim_contract_v3.py tests/unit/test_services/test_provider_reliability.py tests/unit/test_services/test_event_resolution.py tests/unit/test_services/test_event_identity.py tests/unit/test_services/test_recommendation_ledger.py tests/unit/test_services/test_store_migrations.py tests/unit/test_scripts/test_alert_outbox.py tests/unit/test_scripts/test_alert_delivery.py`
  - `279 passed in 2.62s`
- `git diff --check`: passed with no output.
