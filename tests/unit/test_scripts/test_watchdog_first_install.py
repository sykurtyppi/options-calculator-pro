"""Activation ownership governs first-run evidence-cycle alerts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import watch_daily_evidence_cycle as watchdog


def _write_activation(path: Path, value: str | None) -> None:
    if value is not None:
        path.write_text(value + "\n", encoding="utf-8")


def _run_watchdog(
    *,
    monkeypatch,
    tmp_path: Path,
    capsys,
    instant: str,
    activation: str | None,
    explicit_date: str | None = None,
    resolver_issues: list[dict] | None = None,
) -> tuple[int, dict, list[dict]]:
    fixed = datetime.fromisoformat(instant)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz is None else fixed.astimezone(tz)

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    activation_path = tmp_path / "watchdog_activated_at"
    _write_activation(activation_path, activation)
    args = [
        "watchdog",
        "--report-dir",
        str(report_dir),
        "--log-path",
        str(tmp_path / "cycle.log"),
        "--state-path",
        str(tmp_path / "alert-state.json"),
        "--activation-path",
        str(activation_path),
        "--scheduler-timezone",
        "UTC",
        "--no-fetch",
        "--dry-run-alert",
    ]
    if explicit_date:
        args.extend(["--date", explicit_date])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(watchdog, "datetime", Clock)
    monkeypatch.setattr(
        watchdog,
        "build_candidate_exit_resolver_health",
        lambda **kw: {"issues": resolver_issues or [], "summary": {}},
    )
    monkeypatch.setattr(
        watchdog,
        "build_launchd_job_freshness",
        lambda **kw: {"issues": [], "summary": {}},
    )
    monkeypatch.setattr(
        watchdog,
        "build_evidence_funnel_health",
        lambda: {"issues": [], "summary": {}},
    )
    monkeypatch.setattr(
        watchdog,
        "build_provider_budget_health",
        lambda: {"issues": [], "summary": {}},
    )
    monkeypatch.setattr(watchdog, "deployment_drift", lambda **kw: {})
    monkeypatch.setattr(
        watchdog,
        "build_code_identity_health",
        lambda *a, **kw: {"issues": [], "summary": {}},
    )
    sent: list[dict] = []
    def fake_alert(status, **kw):
        if not status.get("ok"):
            sent.append(status)
        return {"attempted": False, "reason": "test"}

    monkeypatch.setattr(watchdog, "maybe_send_watchdog_alert", fake_alert)

    exit_code = watchdog.main()
    payload = json.loads(capsys.readouterr().out)
    return exit_code, payload, sent


@pytest.mark.parametrize(
    "instant,activation,expected_date",
    [
        # Installed before today's first cycle; yesterday is not owned.
        ("2026-11-10T14:35:00+00:00", "2026-11-10T14:35:00+00:00", "2026-11-09"),
        # Weekend install does not own Friday's cycle.
        ("2026-11-07T22:15:00+00:00", "2026-11-07T20:00:00+00:00", "2026-11-06"),
        # Thanksgiving install does not own Wednesday's cycle.
        ("2026-11-26T22:15:00+00:00", "2026-11-26T20:00:00+00:00", "2026-11-25"),
        # Installed after today's 16:45 cycle had already fired.
        ("2026-11-10T22:16:00+00:00", "2026-11-10T20:00:00+00:00", "2026-11-10"),
    ],
)
def test_activation_after_selected_cycle_defers_alert(
    monkeypatch, tmp_path, capsys, instant, activation, expected_date,
):
    exit_code, payload, sent = _run_watchdog(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        capsys=capsys,
        instant=instant,
        activation=activation,
    )

    status = payload["watchdog"]
    assert exit_code == 0
    assert status["ok"] is True
    assert status["status"] == "activation_deferred"
    assert status["expected_date"] == expected_date
    assert status["activation_at"] == activation
    assert sent == []


def test_activation_deferral_does_not_hide_other_operational_failures(
    monkeypatch, tmp_path, capsys,
):
    exit_code, payload, sent = _run_watchdog(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        capsys=capsys,
        instant="2026-11-10T14:35:00+00:00",
        activation="2026-11-10T14:35:00+00:00",
        resolver_issues=[
            {"severity": "FAIL", "alertable": True, "message": "resolver failed"}
        ],
    )

    assert exit_code == 2
    assert payload["watchdog"]["status"] == "activation_deferred"
    assert "resolver failed" in payload["watchdog"]["errors"]
    assert len(sent) == 1


@pytest.mark.parametrize(
    "instant,activation,explicit_date",
    [
        # The installation was active before the first cycle, which never ran.
        ("2026-11-11T06:15:00+00:00", "2026-11-10T14:35:00+00:00", None),
        # Restart/data loss must not reset ownership of an established install.
        ("2026-11-10T14:35:00+00:00", "2026-11-09T12:00:00+00:00", None),
        # Installed before today's cycle, so today's missing result is owned.
        ("2026-11-10T22:16:00+00:00", "2026-11-10T14:35:00+00:00", None),
        # Missing, corrupt, or future markers fail closed rather than hiding a miss.
        ("2026-11-11T06:15:00+00:00", None, None),
        ("2026-11-11T06:15:00+00:00", "not-a-timestamp", None),
        ("2026-11-11T06:15:00+00:00", "2026-11-12T00:00:00+00:00", None),
        # Explicit historical checks are never suppressed by activation state.
        ("2026-11-10T14:35:00+00:00", "2026-11-10T14:35:00+00:00", "2026-11-09"),
    ],
)
def test_owned_or_unverifiable_missing_cycle_alerts(
    monkeypatch, tmp_path, capsys, instant, activation, explicit_date,
):
    exit_code, payload, sent = _run_watchdog(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        capsys=capsys,
        instant=instant,
        activation=activation,
        explicit_date=explicit_date,
    )

    assert exit_code == 2
    assert payload["watchdog"]["ok"] is False
    assert any("Missing evidence report" in error for error in payload["watchdog"]["errors"])
    assert len(sent) == 1


@pytest.mark.parametrize(
    "now,activation,session_date,expected",
    [
        # Friday 16:15 EST is before Friday's 16:45 EST cycle, even when
        # evaluated after the spring-forward transition.
        (
            datetime(2026, 3, 9, 6, 15, tzinfo=timezone.utc),
            datetime(2026, 3, 6, 21, 15, tzinfo=timezone.utc),
            date(2026, 3, 6),
            False,
        ),
        # Friday 17:00 EDT is after Friday's 16:45 EDT cycle, even when
        # evaluated after the fall-back transition.
        (
            datetime(2026, 11, 2, 6, 15, tzinfo=timezone.utc),
            datetime(2026, 10, 30, 21, 0, tzinfo=timezone.utc),
            date(2026, 10, 30),
            True,
        ),
        # An activation attempt stamped exactly at the cycle deadline occurs
        # before launchctl load, so that session was not yet owned.
        (
            datetime(2026, 11, 2, 6, 15, tzinfo=timezone.utc),
            datetime(2026, 10, 30, 20, 45, tzinfo=timezone.utc),
            date(2026, 10, 30),
            True,
        ),
    ],
)
def test_activation_ownership_uses_session_dates_timezone(
    now, activation, session_date, expected,
):
    assert watchdog._activation_defers_session(
        activated_at=activation,
        expected_date=session_date,
        now=now,
        scheduler_tz=ZoneInfo("America/New_York"),
        cycle_hour=16,
        cycle_minute=45,
    ) is expected


@pytest.mark.parametrize("configured", ["Not/A_Real_Zone", "/tmp/not-a-zone"])
def test_invalid_explicit_scheduler_timezone_fails_closed(configured):
    assert watchdog._scheduler_timezone(configured) is None


def _fake_launchctl(fake_bin: Path, body: str) -> None:
    launchctl = fake_bin / "launchctl"
    launchctl.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    launchctl.chmod(0o755)


def _installer_env(home: Path, fake_bin: Path) -> dict[str, str]:
    return {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }


def test_installer_creates_activation_after_cycle_load_and_preserves_it(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    installer = repo / "scripts" / "automation" / "install_launchd_jobs.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    marker = home / ".options_calculator_pro" / "state" / "watchdog_activated_at"
    _fake_launchctl(
        fake_bin,
        f'''case "$1:$2" in
  load:*evidence-cycle.plist) test ! -e "{marker}" || exit 41 ;;
  load:*evidence-watchdog.plist) test -e "{marker}" || exit 42 ;;
esac
exit 0
''',
    )
    env = _installer_env(home, fake_bin)

    first = subprocess.run(
        ["bash", str(installer)], cwd=repo, env=env, capture_output=True, text=True, timeout=30
    )
    assert first.returncode == 0, first.stderr
    first_value = marker.read_text(encoding="utf-8").strip()
    parsed = datetime.fromisoformat(first_value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None

    marker.write_text("2026-01-02T03:04:05Z\n", encoding="utf-8")
    _fake_launchctl(fake_bin, "exit 0\n")
    second = subprocess.run(
        ["bash", str(installer)], cwd=repo, env=env, capture_output=True, text=True, timeout=30
    )
    assert second.returncode == 0, second.stderr
    assert marker.read_text(encoding="utf-8") == "2026-01-02T03:04:05Z\n"


def test_installer_migrates_established_install_without_resetting_ownership(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    installer = repo / "scripts" / "automation" / "install_launchd_jobs.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_launchctl(fake_bin, "exit 0\n")
    home = tmp_path / "home"
    agents = home / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    (agents / "com.optionscalculator.evidence-cycle.plist").write_text(
        "established", encoding="utf-8"
    )

    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo,
        env=_installer_env(home, fake_bin),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    marker = home / ".options_calculator_pro" / "state" / "watchdog_activated_at"
    assert marker.read_text(encoding="utf-8") == "1970-01-01T00:00:00Z\n"


def test_installer_does_not_activate_when_evidence_cycle_load_fails(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    installer = repo / "scripts" / "automation" / "install_launchd_jobs.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_launchctl(
        fake_bin,
        '''case "$1:$2" in
  load:*evidence-cycle.plist) exit 23 ;;
esac
exit 0
''',
    )
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo,
        env=_installer_env(home, fake_bin),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 23
    marker = home / ".options_calculator_pro" / "state" / "watchdog_activated_at"
    assert not marker.exists()

    # A retry after the partial first install is still a fresh installation;
    # the rendered plist left by the failed run must not trigger migration.
    _fake_launchctl(fake_bin, "exit 0\n")
    retry = subprocess.run(
        ["bash", str(installer)],
        cwd=repo,
        env=_installer_env(home, fake_bin),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert retry.returncode == 0, retry.stderr
    activated_at = marker.read_text(encoding="utf-8").strip()
    assert activated_at != "1970-01-01T00:00:00Z"
    assert datetime.fromisoformat(activated_at.replace("Z", "+00:00")).tzinfo is not None


def test_installer_stops_before_watchdog_when_activation_publish_fails(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    installer = repo / "scripts" / "automation" / "install_launchd_jobs.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    _fake_launchctl(fake_bin, f'printf "%s\\n" "$2" >> "{launchctl_log}"\nexit 0\n')
    fake_ln = fake_bin / "ln"
    fake_ln.write_text("#!/bin/sh\nexit 19\n", encoding="utf-8")
    fake_ln.chmod(0o755)
    fake_date = fake_bin / "date"
    fake_date.write_text(
        "#!/bin/sh\nprintf '%s\\n' '2026-11-10T16:40:00Z'\n", encoding="utf-8"
    )
    fake_date.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo,
        env=_installer_env(home, fake_bin),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    marker = home / ".options_calculator_pro" / "state" / "watchdog_activated_at"
    assert not marker.exists()
    pending_activation = (
        home / ".options_calculator_pro" / "state" / "watchdog_activation_pending_at"
    )
    assert pending_activation.read_text(encoding="utf-8") == "2026-11-10T16:40:00Z\n"
    loaded = launchctl_log.read_text(encoding="utf-8")
    assert "com.optionscalculator.evidence-cycle.plist" in loaded
    assert "com.optionscalculator.evidence-watchdog.plist" not in loaded

    # A retry after the evidence job may already have loaded must preserve the
    # first attempt's conservative ownership time, not publish the later retry.
    fake_ln.write_text('#!/bin/sh\nexec /bin/ln "$@"\n', encoding="utf-8")
    fake_date.write_text(
        "#!/bin/sh\nprintf '%s\\n' '2026-11-10T18:00:00Z'\n", encoding="utf-8"
    )
    retry = subprocess.run(
        ["bash", str(installer)],
        cwd=repo,
        env=_installer_env(home, fake_bin),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert retry.returncode == 0, retry.stderr
    assert marker.read_text(encoding="utf-8") == "2026-11-10T16:40:00Z\n"
    assert not pending_activation.exists()


def test_installer_refuses_overlapping_transition(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    installer = repo / "scripts" / "automation" / "install_launchd_jobs.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    _fake_launchctl(fake_bin, f'printf "%s\\n" "$2" >> "{launchctl_log}"\nexit 0\n')
    home = tmp_path / "home"
    lock_dir = home / ".options_calculator_pro" / "state" / "watchdog_install_lock"
    lock_dir.mkdir(parents=True)

    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo,
        env=_installer_env(home, fake_bin),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "already in progress" in result.stderr
    assert not launchctl_log.exists()
    assert lock_dir.is_dir()


def test_installer_refuses_stale_lock_without_racy_reclamation(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    installer = repo / "scripts" / "automation" / "install_launchd_jobs.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_launchctl(fake_bin, "exit 0\n")
    home = tmp_path / "home"
    lock_dir = home / ".options_calculator_pro" / "state" / "watchdog_install_lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "pid").write_text("99999999\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo,
        env=_installer_env(home, fake_bin),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "already in progress" in result.stderr
    assert lock_dir.is_dir()


def test_installer_terminates_after_signal_and_releases_lock(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    installer = repo / "scripts" / "automation" / "install_launchd_jobs.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    load_started = tmp_path / "load-started"
    _fake_launchctl(
        fake_bin,
        f'printf "%s %s\\n" "$1" "$2" >> "{launchctl_log}"\n'
        f'if [ "$1" = "load" ] && [ ! -e "{load_started}" ]; then '
        f'touch "{load_started}"; sleep 1; fi\n'
        "exit 0\n",
    )
    home = tmp_path / "home"
    home.mkdir()

    process = subprocess.Popen(
        ["bash", str(installer)],
        cwd=repo,
        env=_installer_env(home, fake_bin),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not load_started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert load_started.exists()

    process.terminate()
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode != 0, (stdout, stderr)
    lock_dir = home / ".options_calculator_pro" / "state" / "watchdog_install_lock"
    assert not lock_dir.exists()
    operations = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert len(operations) <= 2
