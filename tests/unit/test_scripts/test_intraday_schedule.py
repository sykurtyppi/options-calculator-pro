"""The evidence cycle prices live quotes, so it runs during the NYSE session only.

Every scheduled run used to fire at 21:30 local (after the close on a GMT
Mac) and on weekends: yfinance quotes then were stale or zero-bid, and 11 of
14 no-eligible-structure stops failed the 12% spread gate on liquid names.
"""
from __future__ import annotations

import json
import plistlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import run_evidence_cycle as cycle
from services import evidence_health as health
from services.automation_watchdog import EvidenceWatchdogConfig, build_evidence_watchdog_status
from services.code_identity import build_code_identity_health
from services.market_calendar import EXCHANGE_TZ, days_since_previous_session, in_regular_session

REPO = Path(__file__).resolve().parents[3]
PLIST = REPO / "scripts" / "automation" / "com.optionscalculator.evidence-cycle.plist"


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# ── calendar helpers ────────────────────────────────────────────────────────


@pytest.mark.parametrize("moment,expected", [
    (_utc(2026, 11, 4, 16, 45), True),    # Wednesday 11:45 EST
    (_utc(2026, 11, 4, 14, 29), False),   # 09:29 EST, before the open
    (_utc(2026, 11, 4, 14, 30), True),    # 09:30 EST, the open
    (_utc(2026, 11, 4, 21, 0), False),    # 16:00 EST, the close
    (_utc(2026, 11, 4, 21, 30), False),   # the old schedule: 16:30 EST
    (_utc(2026, 10, 24, 16, 45), False),  # Saturday
    (_utc(2026, 11, 26, 16, 45), False),  # Thanksgiving
    (_utc(2026, 11, 27, 17, 45), True),   # day after Thanksgiving, 12:45 EST
    (_utc(2026, 11, 27, 18, 0), False),   # 13:00 EST early close
])
def test_in_regular_session(moment: datetime, expected: bool) -> None:
    assert in_regular_session(moment) is expected


def test_in_regular_session_needs_an_aware_time() -> None:
    with pytest.raises(ValueError):
        in_regular_session(datetime(2026, 11, 4, 12, 0))


@pytest.mark.parametrize("day,expected", [
    (date(2026, 11, 4), 1),   # Wednesday
    (date(2026, 11, 2), 3),   # Monday
    (date(2026, 11, 1), 2),   # Sunday
    (date(2026, 9, 8), 4),    # Tuesday after Labor Day
])
def test_days_since_previous_session(day: date, expected: int) -> None:
    assert days_since_previous_session(day) == expected


# ── the schedule ────────────────────────────────────────────────────────────


def test_schedule_is_inside_the_session_before_early_closes_in_both_dst_regimes() -> None:
    interval = plistlib.loads(PLIST.read_bytes())["StartCalendarInterval"]
    assert interval == {"Hour": 16, "Minute": 45}
    # On the GMT Mac: a summer (EDT) and a winter (EST) trading day.
    for day in (date(2026, 7, 15), date(2026, 12, 9)):
        fire = datetime(day.year, day.month, day.day, interval["Hour"], interval["Minute"], tzinfo=timezone.utc)
        assert in_regular_session(fire)
        assert fire.astimezone(EXCHANGE_TZ).hour < 13  # still open on an early-close day


# ── the guard ───────────────────────────────────────────────────────────────


def test_skip_reason() -> None:
    assert cycle.session_skip_reason(_utc(2026, 11, 4, 16, 45)) is None
    status, reason = cycle.session_skip_reason(_utc(2026, 10, 24, 16, 45))
    assert status == "skipped" and "market closed on 2026-10-24" in reason
    status, reason = cycle.session_skip_reason(_utc(2026, 11, 4, 21, 30))
    assert status == "fail" and "16:30 ET" in reason and "--allow-outside-session" in reason


@pytest.fixture
def ran(monkeypatch):
    calls: list[dict] = []

    def fake_cycle(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(cycle, "run_evidence_cycle", fake_cycle)
    return calls


def _log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_a_closed_market_day_is_skipped_without_running(monkeypatch, tmp_path: Path, ran, capsys) -> None:
    monkeypatch.setattr(cycle, "session_skip_reason", lambda now: ("skipped", "market closed on 2026-10-24"))
    log = tmp_path / "runs.jsonl"
    assert cycle.main(["--structured-run-log", str(log)]) == 0
    assert ran == []
    [event] = _log(log)
    assert event["status"] == "skipped" and event["error"] == "market closed on 2026-10-24"
    assert json.loads(capsys.readouterr().out)["status"] == "skipped"


def test_a_trading_day_outside_the_session_fails_visibly(monkeypatch, tmp_path: Path, ran) -> None:
    monkeypatch.setattr(cycle, "session_skip_reason", lambda now: ("fail", "outside the NYSE regular session"))
    log = tmp_path / "runs.jsonl"
    assert cycle.main(["--structured-run-log", str(log)]) == cycle.EXIT_OUTSIDE_SESSION == 3
    assert ran == []
    assert _log(log)[0]["status"] == "fail"


@pytest.mark.parametrize("flags", [["--dry-run"], ["--date", "2026-10-24"], ["--allow-outside-session"]])
def test_manual_runs_are_not_held_to_the_session(monkeypatch, tmp_path: Path, ran, flags) -> None:
    def refuse(now):
        raise AssertionError("the session guard must not run")
    monkeypatch.setattr(cycle, "session_skip_reason", refuse)
    assert cycle.main([*flags, "--structured-run-log", str(tmp_path / "runs.jsonl")]) == 0
    assert len(ran) == 1


def test_a_scheduled_run_in_session_runs(monkeypatch, tmp_path: Path, ran) -> None:
    monkeypatch.setattr(cycle, "session_skip_reason", lambda now: None)
    assert cycle.main(["--structured-run-log", str(tmp_path / "runs.jsonl")]) == 0
    assert len(ran) == 1


# ── watchdog and health on closed days ──────────────────────────────────────


@pytest.mark.parametrize("day", [date(2026, 10, 24), date(2026, 11, 26)])  # Saturday, Thanksgiving
def test_watchdog_expects_no_report_when_the_market_is_closed(tmp_path: Path, day: date) -> None:
    status = build_evidence_watchdog_status(
        config=EvidenceWatchdogConfig(expected_date=day, report_dir=tmp_path, log_path=tmp_path / "none.log"),
        now=_utc(day.year, day.month, day.day, 23, 0))
    assert status["ok"] is True and status["status"] == "market_closed" and status["errors"] == []
    assert status["report_path"].endswith(f"evidence_report_{day.isoformat()}.json")


def test_watchdog_still_fails_a_trading_day_without_a_report(tmp_path: Path) -> None:
    status = build_evidence_watchdog_status(
        config=EvidenceWatchdogConfig(expected_date=date(2026, 11, 4), report_dir=tmp_path,
                                      log_path=tmp_path / "none.log"),
        now=_utc(2026, 11, 4, 23, 0))
    assert status["ok"] is False and any("Missing evidence report" in e for e in status["errors"])


def test_health_expects_no_daily_report_when_the_market_is_closed(tmp_path: Path) -> None:
    cfg = health.EvidenceHealthConfig(expected_date=date(2026, 10, 24), report_dir=tmp_path)
    result = health._check_daily_report(cfg, _utc(2026, 10, 24, 23, 0))
    assert result["issues"] == [] and result["summary"]["market_closed"] is True
    trading = health.EvidenceHealthConfig(expected_date=date(2026, 11, 4), report_dir=tmp_path)
    assert any(i["severity"] == "FAIL" for i in health._check_daily_report(trading, _utc(2026, 11, 4, 23, 0))["issues"])


def _run_log(tmp_path: Path, end: datetime) -> Path:
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps({"status": "success", "start_time": end.isoformat(), "end_time": end.isoformat()}) + "\n")
    return path


@pytest.mark.parametrize("expected,now,stale", [
    # Friday's 16:50 run, checked Monday noon: 67h old, inside 36h + 48h.
    (date(2026, 11, 2), _utc(2026, 11, 2, 12, 0), False),
    # The same age on a Wednesday: stale.
    (date(2026, 11, 4), _utc(2026, 11, 4, 12, 0), True),
])
def test_run_log_staleness_allows_for_closed_days(tmp_path: Path, expected: date, now: datetime, stale: bool) -> None:
    log = _run_log(tmp_path, now - timedelta(hours=67))
    cfg = health.EvidenceHealthConfig(expected_date=expected, structured_run_log=log)
    issues = health._check_structured_run_log(cfg, now)["issues"]
    assert any("stale" in i["message"] for i in issues) is stale


@pytest.mark.parametrize("day,hours", [(date(2026, 11, 4), 0), (date(2026, 11, 2), 48), (date(2026, 9, 8), 72)])
def test_closed_market_allowance(day: date, hours: float) -> None:
    assert health._closed_market_allowance_hours(day) == hours


def test_code_identity_ignores_the_report_on_closed_days() -> None:
    drift = {"git_sha": "a" * 40, "dirty": False, "upstream": "origin/main", "upstream_sha": "a" * 40,
             "behind": 0, "ahead": 0, "fetched": True, "fetch_error": None, "error": None}
    assert build_code_identity_health(drift, None, report_expected=False)["issues"] == []
    assert build_code_identity_health(drift, None)["issues"]  # a trading day still wants the stamp


@pytest.mark.parametrize("expected,now,stale", [
    (date(2026, 11, 2), _utc(2026, 11, 2, 12, 0), False),  # 80h old on a Monday: inside 48h + 48h
    (date(2026, 11, 4), _utc(2026, 11, 4, 12, 0), True),   # 80h old on a Wednesday: stale
])
def test_telemetry_staleness_allows_for_closed_days(tmp_path: Path, expected: date, now: datetime,
                                                    stale: bool) -> None:
    import sqlite3

    store = tmp_path / "telemetry.sqlite"
    conn = sqlite3.connect(store)
    conn.execute("CREATE TABLE provider_telemetry (request_timestamp TEXT)")
    conn.execute("INSERT INTO provider_telemetry VALUES (?)", ((now - timedelta(hours=80)).isoformat(),))
    conn.commit()
    conn.close()
    cfg = health.EvidenceHealthConfig(expected_date=expected, telemetry_store_path=store)
    issues = health._check_provider_telemetry(cfg, now)["issues"]
    assert any("stale" in i["message"] for i in issues) is stale
