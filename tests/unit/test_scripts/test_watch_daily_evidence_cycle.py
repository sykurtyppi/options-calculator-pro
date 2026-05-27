"""Tests for scripts/watch_daily_evidence_cycle.py — the iMessage
alert dispatcher that combines daily-cycle status with PR-AE
resolver health.

Ops-AE C1c (Codex P1 audit fix) added a dedicated helper
``_build_combined_watchdog_status`` to make the alert-decision
logic directly unit-testable without spawning the CLI. The tests
below pin two contracts:

  1. Resolver FAIL  → combined ok=False, alert fires.
  2. Resolver WARN  → combined ok=False, alert fires (escalated
     per the Ops-AE taxonomy; was the P1 bug).

Plus a regression that daily-cycle warnings remain observation-only
(unchanged from before C1c).
"""
from __future__ import annotations

from scripts.watch_daily_evidence_cycle import _build_combined_watchdog_status


def _make_watchdog_status(*, ok: bool = True, errors=None, warnings=None) -> dict:
    return {
        "ok": ok,
        "expected_date": "2026-05-27",
        "errors": list(errors or []),
        "warnings": list(warnings or []),
        "report_path": "/tmp/report.json",
    }


def _make_resolver_health(issues=None, summary=None) -> dict:
    return {
        "issues": list(issues or []),
        "summary": dict(summary or {}),
    }


def _issue(severity: str, message: str) -> dict:
    return {
        "severity": severity,
        "check": "candidate_exit_resolver",
        "message": message,
        "fix": "some fix",
    }


# ──────────────────────────────────────────────────────────────────────────
# Baseline: no resolver issues → combined ok matches watchdog ok
# ──────────────────────────────────────────────────────────────────────────


def test_combined_status_is_ok_when_both_sides_clean() -> None:
    combined = _build_combined_watchdog_status(
        watchdog_status=_make_watchdog_status(ok=True),
        resolver_health=_make_resolver_health(),
    )
    assert combined["ok"] is True
    assert combined["errors"] == []
    assert combined["warnings"] == []


def test_combined_status_inherits_watchdog_ok_false() -> None:
    combined = _build_combined_watchdog_status(
        watchdog_status=_make_watchdog_status(
            ok=False, errors=["Missing evidence report"]
        ),
        resolver_health=_make_resolver_health(),
    )
    assert combined["ok"] is False
    assert "Missing evidence report" in combined["errors"]


# ──────────────────────────────────────────────────────────────────────────
# Resolver FAIL → combined ok=False (was already correct in C1)
# ──────────────────────────────────────────────────────────────────────────


def test_resolver_fail_flips_combined_ok_to_false() -> None:
    combined = _build_combined_watchdog_status(
        watchdog_status=_make_watchdog_status(ok=True),
        resolver_health=_make_resolver_health(
            issues=[_issue("FAIL", "resolver count_balance_holds: false")],
        ),
    )
    assert combined["ok"] is False
    assert "resolver count_balance_holds: false" in combined["errors"]


# ──────────────────────────────────────────────────────────────────────────
# C1c P1 fix: Resolver WARN ALSO flips combined ok=False
# ──────────────────────────────────────────────────────────────────────────


def test_ops_ae_c1c_resolver_warn_stuck_awaiting_flips_combined_ok_to_false() -> None:
    """REGRESSION (Codex Ops-AE C1c P1): the prior code only counted
    resolver FAILs into ``ok=False``. A stuck-awaiting WARN (>10 days)
    went into ``combined_status['warnings']`` and the alert never
    fired — but the Ops-AE alert taxonomy explicitly listed
    ``days_in_awaiting_state > 10`` as alertable.

    Post-fix: resolver WARN AND FAIL both contribute to ``ok=False``
    and both appear in ``errors`` for the alert message body. Only
    daily-cycle WARNs remain observation-only.
    """
    combined = _build_combined_watchdog_status(
        watchdog_status=_make_watchdog_status(ok=True),
        resolver_health=_make_resolver_health(
            issues=[
                _issue(
                    "WARN",
                    "Candidate exit resolver has row(s) awaiting chain data for 15 day(s).",
                )
            ],
        ),
    )
    assert combined["ok"] is False
    assert any(
        "awaiting chain data" in error for error in combined["errors"]
    ), f"resolver WARN must appear in errors for the alert message body; got {combined['errors']!r}"


def test_ops_ae_c1c_resolver_warn_stale_completion_flips_combined_ok_to_false() -> None:
    """The other documented resolver WARN-alert case: stale launchd
    completion. Same escalation: WARN → contributes to ok=False."""
    combined = _build_combined_watchdog_status(
        watchdog_status=_make_watchdog_status(ok=True),
        resolver_health=_make_resolver_health(
            issues=[
                _issue(
                    "WARN",
                    "Candidate exit resolver completion is stale (28.4h old).",
                )
            ],
        ),
    )
    assert combined["ok"] is False
    assert any("28.4h" in error for error in combined["errors"])


def test_ops_ae_c1c_resolver_warn_missing_log_flips_combined_ok_to_false() -> None:
    """Third documented alertable WARN: launchd log doesn't exist
    after the first scheduled run was expected. Same escalation."""
    combined = _build_combined_watchdog_status(
        watchdog_status=_make_watchdog_status(ok=True),
        resolver_health=_make_resolver_health(
            issues=[
                _issue(
                    "WARN",
                    "Candidate exit resolver launchd log does not exist yet.",
                )
            ],
        ),
    )
    assert combined["ok"] is False
    assert any("does not exist" in error for error in combined["errors"])


# ──────────────────────────────────────────────────────────────────────────
# Regression: daily-cycle WARN stays observation-only
# ──────────────────────────────────────────────────────────────────────────


def test_daily_cycle_warning_stays_observation_only() -> None:
    """The C1c P1 fix is scoped to RESOLVER WARNs. Daily-cycle WARNs
    (e.g. ``latest.json is missing``) come through
    ``build_evidence_watchdog_status`` and end up in
    ``watchdog_status['warnings']``. Those continue to be
    observation-only and do NOT flip combined ok.

    Without this guard, every daily-cycle WARN would page the user
    via iMessage, which was never the intent — the prior model was
    ``WARN observes, FAIL alerts`` for the cycle itself.
    """
    combined = _build_combined_watchdog_status(
        watchdog_status=_make_watchdog_status(
            ok=True, warnings=["latest.json is missing."]
        ),
        resolver_health=_make_resolver_health(),
    )
    assert combined["ok"] is True
    assert "latest.json is missing." in combined["warnings"]
    assert combined["errors"] == []


# ──────────────────────────────────────────────────────────────────────────
# Resolver block surfaces in the summary
# ──────────────────────────────────────────────────────────────────────────


def test_resolver_summary_block_surfaces_in_combined_status() -> None:
    """The resolver health summary (latest_completion_at,
    count_balance_holds, max_days_in_awaiting_state, etc.) must
    appear under ``combined_status['candidate_exit_resolver']`` for
    operator forensics — independent of whether anything alerted."""
    combined = _build_combined_watchdog_status(
        watchdog_status=_make_watchdog_status(ok=True),
        resolver_health=_make_resolver_health(
            summary={
                "latest_completion_at": "2026-05-27T12:31:00+00:00",
                "count_balance_holds": True,
                "max_days_in_awaiting_state": None,
                "still_awaiting_row_count": 0,
                "simulator_error_count": 0,
            },
        ),
    )
    block = combined["candidate_exit_resolver"]
    assert block["latest_completion_at"] == "2026-05-27T12:31:00+00:00"
    assert block["count_balance_holds"] is True


# ──────────────────────────────────────────────────────────────────────────
# Multiple issues: errors aggregate cleanly
# ──────────────────────────────────────────────────────────────────────────


def test_combined_status_aggregates_multiple_resolver_issues() -> None:
    """When the resolver reports both a FAIL and one or more WARNs,
    all of them flow into errors so the alert message includes the
    full context. ``ok`` is False once any resolver issue exists
    (FAIL or WARN)."""
    combined = _build_combined_watchdog_status(
        watchdog_status=_make_watchdog_status(ok=True),
        resolver_health=_make_resolver_health(
            issues=[
                _issue("FAIL", "Recent simulator_error row."),
                _issue("WARN", "Awaiting chain data for 12 day(s)."),
                _issue("WARN", "Completion is stale (28h)."),
            ],
        ),
    )
    assert combined["ok"] is False
    assert len(combined["errors"]) == 3
    assert any("simulator_error" in e for e in combined["errors"])
    assert any("Awaiting chain data" in e for e in combined["errors"])
    assert any("stale" in e for e in combined["errors"])
