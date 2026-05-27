"""Unit tests for scripts/preflight_check.py.

Each check function is tested in isolation with a synthetic filesystem
and subprocess mocks. The aggregator + exit-code logic is tested
separately so a refactor of the check set doesn't have to touch the
exit-code semantics tests.

Conventions:
  - Use tmp_path everywhere; never touch the real ~/.options_calculator_pro/.
  - Subprocess calls are mocked via patch — we trust subprocess itself.
  - Status constants are imported from the script (not re-defined here)
    so any rename of PASS/WARN/FAIL/SKIP fails this file too.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts import preflight_check as pf
from scripts.preflight_check import Status, CheckResult


# ── env_file ──────────────────────────────────────────────────────────────


def test_env_file_missing_is_warn(tmp_path: Path) -> None:
    """Missing .env is operator-actionable but not a hard fail — the
    backend would still boot with defaults (just no provider tokens
    and share-auth off)."""
    r = pf.check_env_file(project_root=tmp_path)
    assert r.status == Status.WARN
    assert ".env not found" in r.message


def test_env_file_present_is_pass(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("MARKETDATA_TOKEN=xxx\n", encoding="utf-8")
    r = pf.check_env_file(project_root=tmp_path)
    assert r.status == Status.PASS
    assert "bytes" in r.message


def test_env_file_unreadable_is_fail(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("x", encoding="utf-8")
    env.chmod(0o000)
    try:
        r = pf.check_env_file(project_root=tmp_path)
    finally:
        env.chmod(0o600)  # cleanup
    assert r.status == Status.FAIL
    assert "not readable" in r.message


# ── python_venv ───────────────────────────────────────────────────────────


def test_python_venv_missing_is_fail(tmp_path: Path) -> None:
    r = pf.check_python_venv(project_root=tmp_path)
    assert r.status == Status.FAIL
    assert ".venv311" in r.message


def test_python_venv_not_executable_is_fail(tmp_path: Path) -> None:
    py = tmp_path / ".venv311" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o600)  # not executable
    r = pf.check_python_venv(project_root=tmp_path)
    assert r.status == Status.FAIL
    assert "not executable" in r.message


def test_python_venv_executable_is_pass(tmp_path: Path) -> None:
    py = tmp_path / ".venv311" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    r = pf.check_python_venv(project_root=tmp_path)
    assert r.status == Status.PASS


# ── frontend_dist ─────────────────────────────────────────────────────────


def test_frontend_dist_missing_is_warn(tmp_path: Path) -> None:
    """Production needs dist; dev mode doesn't. WARN, not FAIL."""
    r = pf.check_frontend_dist(project_root=tmp_path)
    assert r.status == Status.WARN
    assert "npm run build" in r.message


def test_frontend_dist_present_is_pass(tmp_path: Path) -> None:
    dist = tmp_path / "web" / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html/>", encoding="utf-8")
    r = pf.check_frontend_dist(project_root=tmp_path)
    assert r.status == Status.PASS


# ── wrapper_scripts ───────────────────────────────────────────────────────


def test_wrapper_scripts_all_missing_is_fail(tmp_path: Path) -> None:
    (tmp_path / "scripts" / "automation").mkdir(parents=True)
    r = pf.check_wrapper_scripts(project_root=tmp_path)
    assert r.status == Status.FAIL
    assert len(r.details["missing"]) == len(pf._WRAPPER_NAMES)


def test_wrapper_scripts_present_but_not_executable_is_fail(tmp_path: Path) -> None:
    auto = tmp_path / "scripts" / "automation"
    auto.mkdir(parents=True)
    for name in pf._WRAPPER_NAMES:
        (auto / name).write_text("#!/bin/sh\n", encoding="utf-8")
        (auto / name).chmod(0o600)  # not exec
    r = pf.check_wrapper_scripts(project_root=tmp_path)
    assert r.status == Status.FAIL
    assert len(r.details["not_executable"]) == len(pf._WRAPPER_NAMES)


def test_wrapper_scripts_all_present_executable_is_pass(tmp_path: Path) -> None:
    auto = tmp_path / "scripts" / "automation"
    auto.mkdir(parents=True)
    for name in pf._WRAPPER_NAMES:
        p = auto / name
        p.write_text("#!/bin/sh\n", encoding="utf-8")
        p.chmod(0o755)
    r = pf.check_wrapper_scripts(project_root=tmp_path)
    assert r.status == Status.PASS


# ── plist_templates ───────────────────────────────────────────────────────


def test_plist_templates_missing_placeholder_is_fail(tmp_path: Path) -> None:
    """If someone hand-edits a plist and removes __PROJECT_ROOT__, the
    install script's sed substitution silently leaves the wrong path.
    Catch it here."""
    auto = tmp_path / "scripts" / "automation"
    auto.mkdir(parents=True)
    for name in pf._PLIST_NAMES:
        # Valid in every other respect, but missing __PROJECT_ROOT__.
        (auto / name).write_text(
            "<plist><string>__HOME__/foo</string></plist>\n",
            encoding="utf-8",
        )
    r = pf.check_plist_templates(project_root=tmp_path)
    assert r.status == Status.FAIL
    for name in pf._PLIST_NAMES:
        assert r.details["issues"][name] == "missing __PROJECT_ROOT__ placeholder"


def test_plist_templates_all_present_with_placeholders_is_pass(tmp_path: Path) -> None:
    auto = tmp_path / "scripts" / "automation"
    auto.mkdir(parents=True)
    for name in pf._PLIST_NAMES:
        (auto / name).write_text(
            "<plist>\n"
            "<string>__PROJECT_ROOT__/foo</string>\n"
            "<string>__HOME__/bar</string>\n"
            "</plist>\n",
            encoding="utf-8",
        )
    r = pf.check_plist_templates(project_root=tmp_path)
    assert r.status == Status.PASS


# ── launchagents_installed ────────────────────────────────────────────────


def test_launchagents_no_dir_is_skip(tmp_path: Path) -> None:
    r = pf.check_launchagents_installed(launch_agents_dir=tmp_path / "missing")
    assert r.status == Status.SKIP


def test_launchagents_zero_installed_is_skip(tmp_path: Path) -> None:
    """Dir exists but empty (fresh checkout) → SKIP, not FAIL — the
    operator may be running preflight before installing."""
    r = pf.check_launchagents_installed(launch_agents_dir=tmp_path)
    assert r.status == Status.SKIP


def test_launchagents_partial_install_is_warn(tmp_path: Path) -> None:
    """The exact case the user hit after PR #61 merged but jobs hadn't
    been reinstalled yet — 4 of 5 installed. WARN with the missing
    list so the operator knows what to re-install."""
    for name in pf._PLIST_NAMES[:-1]:  # all but the last
        (tmp_path / name).write_text("<plist/>", encoding="utf-8")
    r = pf.check_launchagents_installed(launch_agents_dir=tmp_path)
    assert r.status == Status.WARN
    assert "Partial install" in r.message
    assert r.details["installed_count"] == len(pf._PLIST_NAMES) - 1
    assert r.details["missing"] == [pf._PLIST_NAMES[-1]]


def test_launchagents_all_installed_is_pass(tmp_path: Path) -> None:
    for name in pf._PLIST_NAMES:
        (tmp_path / name).write_text("<plist/>", encoding="utf-8")
    r = pf.check_launchagents_installed(launch_agents_dir=tmp_path)
    assert r.status == Status.PASS


# ── startup_validators (subprocess-based; mocked) ─────────────────────────


def test_startup_validators_pass_on_clean_import(tmp_path: Path) -> None:
    py = tmp_path / ".venv311" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    fake = MagicMock(returncode=0, stdout="preflight_ok\n", stderr="")
    with patch("subprocess.run", return_value=fake):
        r = pf.check_startup_validators(project_root=tmp_path, python_bin=py)
    assert r.status == Status.PASS


def test_startup_validators_fail_surfaces_last_err_line(tmp_path: Path) -> None:
    """The validator's RuntimeError messages are clear (they name the
    misconfigured env var). Surface the last stderr line so the
    operator can copy-paste the fix."""
    py = tmp_path / ".venv311" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    stderr = (
        "Traceback (most recent call last):\n"
        "  File ...\n"
        "RuntimeError: ENABLE_SHARE_AUTH=true requires SESSION_SECRET to be non-default\n"
    )
    fake = MagicMock(returncode=1, stdout="", stderr=stderr)
    with patch("subprocess.run", return_value=fake):
        r = pf.check_startup_validators(project_root=tmp_path, python_bin=py)
    assert r.status == Status.FAIL
    assert "SESSION_SECRET" in r.message


def test_startup_validators_skip_when_python_missing(tmp_path: Path) -> None:
    """If the venv python is missing, the earlier python_venv check
    already FAILed; we SKIP this one rather than emit a duplicate."""
    r = pf.check_startup_validators(
        project_root=tmp_path, python_bin=tmp_path / "nonexistent"
    )
    assert r.status == Status.SKIP


# ── sqlite_stores ─────────────────────────────────────────────────────────


def test_sqlite_stores_all_absent_is_skip(tmp_path: Path) -> None:
    paths = (tmp_path / "a.sqlite", tmp_path / "b.sqlite")
    r = pf.check_sqlite_stores(stores=paths)
    assert r.status == Status.SKIP


def test_sqlite_stores_clean_db_passes_quick_check(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (1)")
    r = pf.check_sqlite_stores(stores=(db,))
    assert r.status == Status.PASS
    assert r.details["checked"][str(db)] == "ok"


def test_sqlite_stores_corrupt_db_is_fail(tmp_path: Path) -> None:
    """Write garbage to a .sqlite file → sqlite3 raises on open. We
    expect FAIL, not crash."""
    db = tmp_path / "ledger.sqlite"
    db.write_bytes(b"not a sqlite file at all")
    r = pf.check_sqlite_stores(stores=(db,))
    assert r.status == Status.FAIL
    assert str(db) in r.details["failures"]


def test_sqlite_stores_mixed_present_absent(tmp_path: Path) -> None:
    """Real production has some stores present (recommendation_ledger,
    after first ledger write) and some absent (telemetry, until first
    provider call). PASS overall — count the ok ones, ignore absent."""
    present = tmp_path / "present.sqlite"
    absent = tmp_path / "absent.sqlite"
    with sqlite3.connect(present) as conn:
        conn.execute("CREATE TABLE t (x INT)")
    r = pf.check_sqlite_stores(stores=(present, absent))
    assert r.status == Status.PASS
    assert r.details["checked"][str(present)] == "ok"
    assert r.details["checked"][str(absent)] == "absent"


# ── log_rotation_dryrun ───────────────────────────────────────────────────


def test_log_rotation_dryrun_pass(tmp_path: Path) -> None:
    py = tmp_path / ".venv311" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    fake = MagicMock(
        returncode=0,
        stdout="  [ok] daily_evidence_cycle_launchd.log: 776.1KB\n"
               "  [ok] candidate_exit_resolver_launchd.log: 1.2KB\n",
        stderr="",
    )
    with patch("subprocess.run", return_value=fake):
        r = pf.check_log_rotation_dryrun(project_root=tmp_path, python_bin=py)
    assert r.status == Status.PASS
    assert r.details["files_scanned"] == 2


def test_log_rotation_dryrun_nonzero_exit_is_fail(tmp_path: Path) -> None:
    py = tmp_path / ".venv311" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    fake = MagicMock(returncode=2, stdout="", stderr="ImportError: ...")
    with patch("subprocess.run", return_value=fake):
        r = pf.check_log_rotation_dryrun(project_root=tmp_path, python_bin=py)
    assert r.status == Status.FAIL


# ── backend_health ────────────────────────────────────────────────────────


def test_backend_health_connection_refused_is_skip() -> None:
    """No backend running → SKIP. Preflight is not responsible for
    starting uvicorn."""
    # 127.0.0.1:1 is virtually guaranteed to be unbound.
    r = pf.check_backend_health(base_url="http://127.0.0.1:1", timeout_seconds=0.5)
    assert r.status == Status.SKIP


def test_backend_health_pass_on_valid_response() -> None:
    """Mock urlopen to return a 200 with the canonical health body."""
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = b'{"status": "ok", "timestamp": "..."}'
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda self, *args: None
    with patch("scripts.preflight_check.urlopen", return_value=fake_resp):
        r = pf.check_backend_health()
    assert r.status == Status.PASS


def test_backend_health_fail_on_wrong_body_shape() -> None:
    """A 200 with the wrong body indicates the URL is being served by
    something other than this app (a reverse proxy quirk, an old
    deployment). Don't paper over it."""
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = b'<html>not this app</html>'
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda self, *args: None
    with patch("scripts.preflight_check.urlopen", return_value=fake_resp):
        r = pf.check_backend_health()
    assert r.status == Status.FAIL


# ── aggregation + exit code ───────────────────────────────────────────────


def _r(status: Status) -> CheckResult:
    return CheckResult(name="x", status=status, message="")


def test_aggregate_exit_code_all_pass_is_zero() -> None:
    assert pf.aggregate_exit_code([_r(Status.PASS), _r(Status.PASS)]) == 0


def test_aggregate_exit_code_skips_count_as_zero() -> None:
    """SKIP must NOT push to a higher exit code — preflight is meant
    to be safe to run on fresh installs."""
    assert pf.aggregate_exit_code([_r(Status.PASS), _r(Status.SKIP)]) == 0


def test_aggregate_exit_code_warn_only_is_one() -> None:
    assert pf.aggregate_exit_code([_r(Status.PASS), _r(Status.WARN)]) == 1


def test_aggregate_exit_code_any_fail_is_two() -> None:
    """A single FAIL dominates — operator must see exit 2 even if
    everything else is PASS."""
    assert pf.aggregate_exit_code(
        [_r(Status.PASS), _r(Status.WARN), _r(Status.FAIL)]
    ) == 2


# ── output formatting ─────────────────────────────────────────────────────


def test_format_json_is_parseable() -> None:
    results = [
        _r(Status.PASS),
        CheckResult(name="y", status=Status.FAIL, message="boom", details={"k": "v"}),
    ]
    payload = json.loads(pf.format_json(results))
    assert payload["summary"]["PASS"] == 1
    assert payload["summary"]["FAIL"] == 1
    # Details survive the round-trip.
    failed = next(c for c in payload["checks"] if c["status"] == "FAIL")
    assert failed["details"] == {"k": "v"}


def test_format_human_includes_all_check_names() -> None:
    results = [_r(Status.PASS), _r(Status.WARN), _r(Status.FAIL)]
    text = pf.format_human(results)
    # Status glyphs visible
    assert "PASS" in text and "WARN" in text and "FAIL" in text
    # Summary line present
    assert "Summary:" in text
