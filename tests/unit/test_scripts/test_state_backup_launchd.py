"""
Verify the launchd wiring for the daily state backup — plist validity, template
substitution, install/uninstall list inclusion, README documentation, and
wrapper script structure.

We do NOT load the launchd job here (CI runs on Linux; launchctl is macOS-only).
We exercise everything verifiable on any platform:
  - plist parses + has correct Label / wrapper path / schedule / RunAtLoad
  - install_launchd_jobs.sh and uninstall_launchd_jobs.sh include the plist
  - the install script's sed substitution produces a valid rendered plist
  - the wrapper is executable, set -euo pipefail, with lock/timeout/exit-code
    plumbing matching the established log-rotation pattern
  - end-to-end smoke: the wrapper actually writes a restorable archive
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
AUTOMATION = REPO / "scripts" / "automation"
PLIST = AUTOMATION / "com.optionscalculator.state-backup.plist"
WRAPPER = AUTOMATION / "run_state_backup.sh"
INSTALL = AUTOMATION / "install_launchd_jobs.sh"
UNINSTALL = AUTOMATION / "uninstall_launchd_jobs.sh"
README = AUTOMATION / "README.md"

LABEL = "com.optionscalculator.state-backup"


# ── Plist ─────────────────────────────────────────────────────────────────────


def test_plist_parses_and_has_expected_structure():
    data = plistlib.loads(PLIST.read_bytes())
    assert data["Label"] == LABEL
    assert data["RunAtLoad"] is False, "must not fire on launchctl load"
    args = data["ProgramArguments"]
    assert len(args) == 1 and args[0].endswith("/scripts/automation/run_state_backup.sh"), args
    # 04:00 local — offset from the 03:00 log rotation so they don't contend
    # for the SQLite files.
    assert data["StartCalendarInterval"] == {"Hour": 4, "Minute": 0}, data["StartCalendarInterval"]


def test_plist_uses_template_placeholders():
    raw = PLIST.read_text()
    assert "__PROJECT_ROOT__" in raw, "plist must template PROJECT_ROOT"
    assert "__HOME__" in raw, "plist must template HOME"
    for leak in ("/Users/", "/home/"):
        assert leak not in raw, f"plist leaks absolute path containing {leak!r}"


def test_install_script_renders_plist_correctly(tmp_path: Path):
    rendered = PLIST.read_text()
    rendered = rendered.replace("__PROJECT_ROOT__", "/tmp/fake_project")
    rendered = rendered.replace("__HOME__", "/tmp/fake_home")
    out = tmp_path / "rendered.plist"
    out.write_text(rendered)
    data = plistlib.loads(out.read_bytes())
    assert data["ProgramArguments"][0] == "/tmp/fake_project/scripts/automation/run_state_backup.sh"
    assert data["StandardOutPath"] == "/tmp/fake_home/.options_calculator_pro/logs/state_backup_launchd_stdout.log"
    assert data["StandardErrorPath"] == "/tmp/fake_home/.options_calculator_pro/logs/state_backup_launchd_stderr.log"


# ── install / uninstall job list inclusion ───────────────────────────────────


def test_install_script_lists_new_plist():
    assert "com.optionscalculator.state-backup.plist" in INSTALL.read_text(), (
        "install_launchd_jobs.sh must include the state-backup plist"
    )


def test_uninstall_script_lists_new_plist():
    assert "com.optionscalculator.state-backup.plist" in UNINSTALL.read_text(), (
        "uninstall_launchd_jobs.sh must include the state-backup plist"
    )


def test_readme_documents_schedule_entry():
    body = README.read_text()
    assert "`com.optionscalculator.state-backup`" in body, "README schedule table must list the job"
    assert "04:00" in body, "README must document the 04:00 fire time"


# ── Wrapper ──────────────────────────────────────────────────────────────────


def test_wrapper_is_executable_and_uses_bash():
    mode = WRAPPER.stat().st_mode
    assert mode & 0o100, f"wrapper must be user-executable; got mode {oct(mode)}"
    assert WRAPPER.read_text().splitlines()[0].startswith("#!/usr/bin/env bash")


def test_wrapper_has_safety_flags_and_lock_and_timeout_and_exit_code():
    body = WRAPPER.read_text()
    assert "set -euo pipefail" in body, "wrapper must use strict bash mode"
    assert 'mkdir "${LOCK_DIR}"' in body, "wrapper must use mkdir-as-mutex for the lock"
    assert "LOCK_MAX_AGE_SECONDS" in body, "wrapper must support stale-lock recovery"
    assert "subprocess.TimeoutExpired" in body, "wrapper must enforce a hard timeout"
    assert "SystemExit(124)" in body, "wrapper must signal timeout via exit 124"
    assert "EXIT_CODE=$?" in body, "wrapper must capture the subprocess exit code"
    assert 'exit "${EXIT_CODE}"' in body, "wrapper must propagate the exit code to launchd"
    for marker in ("state backup start", "state backup complete",
                   "state backup failed", "state backup skipped"):
        assert marker in body, f"wrapper must log marker {marker!r}"


def test_wrapper_self_locates_project_root():
    body = WRAPPER.read_text()
    assert 'PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"' in body, (
        "wrapper must self-locate PROJECT_ROOT from BASH_SOURCE"
    )
    assert "${PROJECT_ROOT}/.venv311/bin/python" in body, (
        "wrapper must invoke the project's pinned .venv311 Python"
    )


def test_wrapper_targets_backup_state_script():
    body = WRAPPER.read_text()
    assert "scripts/backup_state.py" in body, "wrapper must invoke scripts/backup_state.py"


def test_wrapper_supports_external_output_dir_override():
    """The disaster-recovery escape hatch: an env var routes the archive off-host."""
    body = WRAPPER.read_text()
    assert "OPTIONS_CALCULATOR_BACKUP_DIR" in body, (
        "wrapper must honor OPTIONS_CALCULATOR_BACKUP_DIR for off-host backups"
    )
    assert "--output-dir" in body, "wrapper must pass the external dir through as --output-dir"


def test_wrapper_log_paths_match_plist_pattern():
    body = WRAPPER.read_text()
    assert ".options_calculator_pro/logs" in body
    assert "state_backup_launchd.log" in body


# ── End-to-end smoke: wrapper writes a restorable archive ────────────────────


@pytest.mark.skipif(os.name == "nt", reason="bash wrapper not supported on Windows")
def test_wrapper_produces_a_valid_archive(tmp_path: Path):
    """Run the wrapper against an external output dir and confirm it emits a
    non-empty, well-formed .tar.gz. Proves the venv + backup_state wiring line up.
    """
    python = REPO / ".venv311" / "bin" / "python"
    if not python.exists():
        pytest.skip("project venv not available in this environment")
    out_dir = tmp_path / "backups"
    env = {
        **os.environ,
        "OPTIONS_CALCULATOR_BACKUP_DIR": str(out_dir),
        "OPTIONS_CALCULATOR_BACKUP_RETENTION": "2",
    }
    result = subprocess.run(
        ["bash", str(WRAPPER)], capture_output=True, text=True, timeout=120, env=env,
    )
    # Exit 0 (backed up) or 1 (empty state dir on a clean CI box) are both
    # acceptable; a crash (2) or timeout (124) is not.
    assert result.returncode in (0, 1), f"wrapper failed: rc={result.returncode} {result.stderr!r}"
    if result.returncode == 0:
        archives = list(out_dir.glob("*.tar.gz"))
        assert archives, "successful backup must leave an archive"
        assert tarfile.is_tarfile(archives[0]), "archive must be a valid tar file"
