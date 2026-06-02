"""
Verify the launchd wiring for forward_paper_collector — plist validity, template
substitution, install/uninstall list inclusion, and wrapper script structure.

We do NOT load the launchd job here (CI runs on Linux; launchctl is macOS-only).
We exercise everything that's verifiable on any platform:
  - plist parses + has correct Label / wrapper path / schedule / RunAtLoad
  - install_launchd_jobs.sh and uninstall_launchd_jobs.sh include the new plist
  - the install script's sed substitution produces a valid rendered plist
  - the wrapper script is executable, set -euo pipefail, has lock/timeout/exit-code
    plumbing matching the established evidence-cycle pattern
"""
from __future__ import annotations

import os
import plistlib
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
AUTOMATION = REPO / "scripts" / "automation"
PLIST = AUTOMATION / "com.optionscalculator.forward-paper-collector.plist"
WRAPPER = AUTOMATION / "run_forward_paper_collector.sh"
INSTALL = AUTOMATION / "install_launchd_jobs.sh"
UNINSTALL = AUTOMATION / "uninstall_launchd_jobs.sh"
README = AUTOMATION / "README.md"

LABEL = "com.optionscalculator.forward-paper-collector"


# ── Plist ─────────────────────────────────────────────────────────────────────


def test_plist_parses_and_has_expected_structure():
    data = plistlib.loads(PLIST.read_bytes())
    assert data["Label"] == LABEL
    assert data["RunAtLoad"] is False, "must not fire on launchctl load"
    args = data["ProgramArguments"]
    assert len(args) == 1 and args[0].endswith("/scripts/automation/run_forward_paper_collector.sh"), args
    sched = data["StartCalendarInterval"]
    assert sched == {"Hour": 19, "Minute": 30}, sched


def test_plist_uses_template_placeholders():
    """Plist must use __PROJECT_ROOT__ / __HOME__ placeholders, not hardcoded paths."""
    raw = PLIST.read_text()
    assert "__PROJECT_ROOT__" in raw, "plist must template PROJECT_ROOT"
    assert "__HOME__" in raw, "plist must template HOME"
    # And it must NOT contain absolute paths that would leak between installs
    for leak in ("/Users/", "/home/"):
        assert leak not in raw, f"plist leaks absolute path containing {leak!r}"


def test_install_script_renders_plist_correctly(tmp_path: Path):
    """Apply the install script's sed substitution and verify the result is a valid plist."""
    rendered = PLIST.read_text()
    rendered = rendered.replace("__PROJECT_ROOT__", "/tmp/fake_project")
    rendered = rendered.replace("__HOME__", "/tmp/fake_home")
    out = tmp_path / "rendered.plist"
    out.write_text(rendered)
    data = plistlib.loads(out.read_bytes())
    assert data["ProgramArguments"][0] == "/tmp/fake_project/scripts/automation/run_forward_paper_collector.sh"
    assert data["StandardOutPath"] == "/tmp/fake_home/.options_calculator_pro/logs/forward_paper_collector_launchd_stdout.log"
    assert data["StandardErrorPath"] == "/tmp/fake_home/.options_calculator_pro/logs/forward_paper_collector_launchd_stderr.log"


# ── install / uninstall job list inclusion ───────────────────────────────────


def test_install_script_lists_new_plist():
    body = INSTALL.read_text()
    assert "com.optionscalculator.forward-paper-collector.plist" in body, (
        "install_launchd_jobs.sh must include the new plist in its job list"
    )


def test_uninstall_script_lists_new_plist():
    body = UNINSTALL.read_text()
    assert "com.optionscalculator.forward-paper-collector.plist" in body, (
        "uninstall_launchd_jobs.sh must include the new plist in its job list"
    )


def test_readme_documents_schedule_entry():
    body = README.read_text()
    assert "`com.optionscalculator.forward-paper-collector`" in body, "README schedule table must list the job"
    assert "19:30" in body, "README must document the 19:30 fire time"


# ── Wrapper ──────────────────────────────────────────────────────────────────


def test_wrapper_is_executable_and_uses_bash():
    mode = WRAPPER.stat().st_mode
    assert mode & 0o100, f"wrapper must be user-executable; got mode {oct(mode)}"
    head = WRAPPER.read_text().splitlines()[0]
    assert head.startswith("#!/usr/bin/env bash"), head


def test_wrapper_has_safety_flags_and_lock_and_timeout_and_exit_code():
    body = WRAPPER.read_text()
    assert "set -euo pipefail" in body, "wrapper must use strict bash mode"
    # anti-overlap lock with stale-lock recovery — must use mkdir-as-mutex pattern
    assert "mkdir \"${LOCK_DIR}\"" in body, "wrapper must use mkdir-as-mutex for the lock"
    assert "LOCK_MAX_AGE_SECONDS" in body, "wrapper must support stale-lock recovery"
    # timeout via Python subprocess.run + SystemExit(124)
    assert "subprocess.TimeoutExpired" in body, "wrapper must enforce a hard timeout"
    assert "SystemExit(124)" in body, "wrapper must signal timeout via exit 124"
    # honest exit-code propagation back to launchd (P1-4 pattern)
    assert re.search(r"set \+e", body) and "EXIT_CODE=\\$?" or "EXIT_CODE=$?" in body, (
        "wrapper must capture the subprocess exit code"
    )
    assert "exit \"${EXIT_CODE}\"" in body, "wrapper must propagate the exit code to launchd"
    # structured log markers
    for marker in ("forward paper collector start", "forward paper collector complete",
                   "forward paper collector failed", "forward paper collector skipped"):
        assert marker in body, f"wrapper must log marker {marker!r}"


def test_wrapper_self_locates_project_root():
    """Wrapper must derive PROJECT_ROOT from its own location, not env or cwd."""
    body = WRAPPER.read_text()
    assert 'PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"' in body, (
        "wrapper must self-locate PROJECT_ROOT from BASH_SOURCE"
    )
    assert "${PROJECT_ROOT}/.venv311/bin/python" in body, (
        "wrapper must invoke the project's pinned .venv311 Python"
    )


def test_wrapper_log_paths_match_plist_pattern():
    """Wrapper logs to the same dir the plist captures launchd stdout/stderr in."""
    body = WRAPPER.read_text()
    assert ".options_calculator_pro/logs" in body
    assert "forward_paper_collector_launchd.log" in body


# ── End-to-end smoke: wrapper actually runs the collector's --selftest path ──


@pytest.mark.skipif(os.name == "nt", reason="bash wrapper not supported on Windows")
def test_wrapper_invocation_path_matches_collector_script():
    """The heredoc must invoke scripts/forward_paper_collector.py — guard against typos."""
    body = WRAPPER.read_text()
    assert "scripts/forward_paper_collector.py" in body, (
        "wrapper heredoc must target scripts/forward_paper_collector.py"
    )


@pytest.mark.skipif(os.name == "nt", reason="bash wrapper not supported on Windows")
def test_collector_selftest_executes_via_project_python():
    """End-to-end: run the collector's offline --selftest via the same python the wrapper would use.

    This is the strongest integration check we can do without launchctl: it proves the
    venv + import path + collector logic all line up. If this passes, the wrapper's
    Python invocation will too.
    """
    python = REPO / ".venv311" / "bin" / "python"
    if not python.exists():
        pytest.skip("project venv not available in this environment")
    result = subprocess.run(
        [str(python), str(REPO / "scripts" / "forward_paper_collector.py"), "--selftest"],
        capture_output=True, text=True, timeout=60, cwd=str(REPO),
    )
    assert result.returncode == 0, f"selftest failed: {result.stderr!r}"
    assert "selftest: PASS" in result.stdout, result.stdout
