"""The API server's launchd job, and no paths a case-insensitive disk merges.

CI runs on Linux; the Mac that runs the jobs has a case-insensitive
filesystem. Both checks here catch on Linux what would only break there.
"""
from __future__ import annotations

import fnmatch
import os
import plistlib
import shutil
import subprocess
from pathlib import Path

from scripts.rotate_launchd_logs import _LAUNCHD_LOG_GLOBS

REPO = Path(__file__).resolve().parents[3]
AUTOMATION = REPO / "scripts" / "automation"
PLIST = AUTOMATION / "com.optionscalculator.api.plist"
WRAPPER = AUTOMATION / "run_api_server.sh"


def test_api_plist_keeps_the_server_running_and_lets_a_stop_stick() -> None:
    data = plistlib.loads(PLIST.read_bytes())
    assert data["Label"] == "com.optionscalculator.api"
    assert data["ProgramArguments"] == ["__PROJECT_ROOT__/scripts/automation/run_api_server.sh"]
    assert data["RunAtLoad"] is True
    # Restart on failure only: a clean `launchctl stop` must stay stopped.
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ThrottleInterval"] >= 30  # a held port retries, never spins
    raw = PLIST.read_text()
    for leak in ("/Users/", "/home/"):
        assert leak not in raw


def test_api_plist_renders_like_the_other_jobs() -> None:
    rendered = PLIST.read_text().replace("__PROJECT_ROOT__", "/p").replace("__HOME__", "/h")
    data = plistlib.loads(rendered.encode())
    assert data["ProgramArguments"] == ["/p/scripts/automation/run_api_server.sh"]
    assert data["StandardOutPath"] == "/h/.options_calculator_pro/logs/api_server_stdout.log"
    assert data["StandardErrorPath"] == "/h/.options_calculator_pro/logs/api_server_stderr.log"


def test_api_logs_are_outside_the_rotation_that_would_orphan_them() -> None:
    data = plistlib.loads(PLIST.read_bytes())
    for key in ("StandardOutPath", "StandardErrorPath"):
        name = Path(data[key]).name
        assert not any(fnmatch.fnmatch(name, g) for g in _LAUNCHD_LOG_GLOBS), name
    # The scheduled jobs' logs are still rotated: the globs are not just empty.
    backup = plistlib.loads((AUTOMATION / "com.optionscalculator.state-backup.plist").read_bytes())
    assert any(fnmatch.fnmatch(Path(backup["StandardOutPath"]).name, g) for g in _LAUNCHD_LOG_GLOBS)


def test_installer_and_uninstaller_manage_the_api_job() -> None:
    for script in ("install_launchd_jobs.sh", "uninstall_launchd_jobs.sh"):
        assert "com.optionscalculator.api.plist" in (AUTOMATION / script).read_text(), script


def test_readme_documents_the_api_job() -> None:
    body = (AUTOMATION / "README.md").read_text()
    assert "`com.optionscalculator.api`" in body
    assert "kickstart -k" in body  # how to load new code after a pull


def _fake_project(tmp_path: Path) -> Path:
    """A project tree whose venv python records how it was called."""
    root = tmp_path / "proj"
    (root / "scripts" / "automation").mkdir(parents=True)
    shutil.copy2(WRAPPER, root / "scripts" / "automation" / WRAPPER.name)
    py = root / ".venv311" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" > "$CALLS"\n')
    py.chmod(0o755)
    return root


def _run(root: Path, tmp_path: Path, **env: str) -> list[str]:
    calls = tmp_path / "calls.txt"
    base = {k: v for k, v in os.environ.items() if not k.startswith("OPTIONS_CALCULATOR_API_")}
    result = subprocess.run(
        [str(root / "scripts" / "automation" / WRAPPER.name)],
        cwd=tmp_path, env={**base, "CALLS": str(calls), **env},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "api server start" in result.stdout
    return calls.read_text().splitlines()


def test_wrapper_execs_uvicorn_on_localhost_from_the_project_root(tmp_path: Path) -> None:
    assert WRAPPER.stat().st_mode & 0o100
    body = WRAPPER.read_text()
    assert body.splitlines()[0] == "#!/usr/bin/env bash" and "set -euo pipefail" in body
    assert 'exec "${PYTHON_BIN}" -m uvicorn' in body  # launchd supervises uvicorn itself
    root = _fake_project(tmp_path)
    cwd, *args = _run(root, tmp_path)
    assert Path(cwd).resolve() == root.resolve()
    assert args == ["-m", "uvicorn", "web.api.app:app", "--host", "127.0.0.1", "--port", "8000",
                    "--no-access-log"]


def test_wrapper_host_and_port_can_be_overridden(tmp_path: Path) -> None:
    root = _fake_project(tmp_path)
    _cwd, *args = _run(root, tmp_path, OPTIONS_CALCULATOR_API_HOST="0.0.0.0", OPTIONS_CALCULATOR_API_PORT="9001")
    assert args[3:7] == ["--host", "0.0.0.0", "--port", "9001"]


# ── case-insensitive path collisions ───────────────────────────────────────

_SKIP_DIRS = {".git", "node_modules", "dist", "__pycache__", ".pytest_cache"}


def case_collisions(root: Path) -> list[list[str]]:
    """Paths under *root* that are distinct here but one path on macOS."""
    seen: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".venv")]
        for name in dirnames + filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            seen.setdefault(rel.lower(), []).append(rel)
    return sorted(sorted(group) for group in seen.values() if len(group) > 1)


def test_case_collision_finder_sees_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "Readme.md").write_text("")
    (tmp_path / "a" / "README.md").write_text("")
    (tmp_path / "Docs").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "X").write_text("")
    (tmp_path / "node_modules" / "x").write_text("")
    assert case_collisions(tmp_path) == [["Docs", "docs"], ["a/README.md", "a/Readme.md"]]


def test_no_two_repo_paths_differ_only_by_case() -> None:
    assert case_collisions(REPO) == []
