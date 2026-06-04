"""Regression guard for default singleton-store isolation.

The full suite once produced three SQLite "attempt to write a readonly database"
failures because production code (run_daily_cycle, evidence_report, ...) reaches
for the default BaselineEvidenceStore / RecommendationLedger singletons whenever
a store is not threaded through explicitly. Under conftest.py's HOME redirect
those fell through to repo-local .pytest_home/.options_calculator_pro/* WAL DBs,
which fail on any host where that path is read-only.

The autouse `_isolate_default_singleton_stores` fixture (tests/conftest.py)
redirects those defaults to a per-test temp dir. These tests fail loudly if that
isolation is removed or stops working — so a future test that intends temp state
can never silently start writing into the repo-local .pytest_home again.
"""
from __future__ import annotations

from pathlib import Path

import services.baseline_evidence_store as bes
import services.recommendation_ledger as rl


def _assert_off_repo_home(path: Path, what: str) -> None:
    text = str(path)
    assert ".pytest_home" not in text, (
        f"{what} resolves to the repo-local .pytest_home ({text!r}); the default "
        "singleton-store isolation fixture is not active. Tests must not write "
        "WAL SQLite into the checkout."
    )


def test_default_store_constants_are_redirected_off_repo_home() -> None:
    _assert_off_repo_home(bes._DEFAULT_STORE, "BaselineEvidenceStore default path")
    _assert_off_repo_home(rl._DEFAULT_LEDGER, "RecommendationLedger default path")


def test_lazily_created_singletons_land_in_temp_not_repo_home() -> None:
    # Exercise the exact fall-through path production code uses.
    baseline = bes.get_baseline_evidence_store()
    ledger = rl.get_recommendation_ledger()

    _assert_off_repo_home(baseline._path, "lazily-created baseline store")
    _assert_off_repo_home(ledger._path, "lazily-created recommendation ledger")


def test_singletons_are_reset_between_tests() -> None:
    # The isolation fixture nulls the module globals around each test, so a fresh
    # getter call rebuilds the singleton at this test's redirected temp path
    # rather than inheriting another test's handle.
    baseline = bes.get_baseline_evidence_store()
    assert bes._store is baseline  # cached within the test
    _assert_off_repo_home(baseline._path, "baseline singleton (this test)")
