#!/usr/bin/env python3
"""Pre-deployment / pre-start smoke check.

Reads the operator's current state and verifies the things that
would otherwise fail at runtime or, worse, fail silently. Designed to
be run **before** flipping ``ENABLE_SHARE_AUTH=true``, before
installing the launchd jobs, and as a sanity sweep before any
non-trivial restart.

Codex scope (PR #62 follow-up): keep it small. This is "am I safe to
deploy/run?", not a giant validator. The runbook
(``docs/DEPLOYMENT.md``) tells a human what to verify; this script
turns the highest-risk parts into executable checks.

Checks (all read-only, none mutate state):

  1. .env file exists and is readable.
  2. ``${PROJECT_ROOT}/.venv311/bin/python`` is present and executable.
  3. Frontend ``web/frontend/dist/`` exists (WARN if missing — dev mode
     is fine, production-same-origin needs the built bundle).
  4. All 5 launchd wrapper scripts (``scripts/automation/run_*.sh``)
     exist and are executable.
  5. All 5 plist templates exist and contain the placeholders the
     install script substitutes.
  6. LaunchAgents installed under ``~/Library/LaunchAgents`` — PASS if
     all 5; WARN if partial; SKIP if none (likely a fresh checkout or
     a dev box).
  7. ``web.api.app`` imports cleanly — exercises the PR #59 startup
     validators (SESSION_SECRET ≥24 chars, ALLOWED_ORIGINS not '*' /
     'null', Secure-cookie posture, etc.). Run in a subprocess so the
     parent process doesn't get polluted by the import.
  8. Known SQLite stores under ``~/.options_calculator_pro/`` pass
     ``PRAGMA quick_check`` if they exist (SKIP if not yet).
  9. ``scripts/rotate_launchd_logs.py --dry-run`` exits 0.
 10. Optional ``GET /api/health`` if a backend is already running —
     SKIP if connection refused (we never start uvicorn here; it's
     the operator's job).

Exit codes (match the watchdog convention so this can drop into
existing CI):

  0 — all checks PASS or SKIP
  1 — at least one WARN, no FAIL
  2 — at least one FAIL

Output:

  Human-readable summary by default. ``--json`` emits a structured
  payload suitable for CI ingestion.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HOME = Path.home()
_LAUNCH_AGENTS = _HOME / "Library" / "LaunchAgents"
_OPTIONS_HOME = _HOME / ".options_calculator_pro"

# Canonical SQLite stores. Each is a service-owned data file; if it
# exists, it should pass quick_check. Missing → SKIP (fresh install).
_KNOWN_SQLITE_STORES: tuple[Path, ...] = (
    _OPTIONS_HOME / "recommendations" / "recommendation_ledger.sqlite",
    _OPTIONS_HOME / "outcomes" / "outcome_store.sqlite",
    _OPTIONS_HOME / "evidence" / "baseline_evidence.sqlite",
    _OPTIONS_HOME / "telemetry" / "provider_telemetry.sqlite",
)

# The full set of launchd plists this project ships. The install
# script (``scripts/automation/install_launchd_jobs.sh``) and
# uninstall script must stay in sync with this list — if you add a
# new launchd job to the install script, add it here too or the
# preflight will silently ignore it.
_PLIST_NAMES: tuple[str, ...] = (
    "com.optionscalculator.candidate-exit-resolver.plist",
    "com.optionscalculator.evidence-cycle.plist",
    "com.optionscalculator.evidence-watchdog.plist",
    "com.optionscalculator.weekly-evidence-report.plist",
    "com.optionscalculator.log-rotation.plist",
)

_WRAPPER_NAMES: tuple[str, ...] = (
    "run_candidate_exit_resolver.sh",
    "run_daily_evidence_cycle.sh",
    "run_daily_evidence_watchdog.sh",
    "run_weekly_evidence_report.sh",
    "run_launchd_log_rotation.sh",
)


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ── Individual checks ────────────────────────────────────────────────────


def check_env_file(project_root: Path = _PROJECT_ROOT) -> CheckResult:
    """The repo's ``.env`` is loaded by every entrypoint via
    ``python-dotenv``. Missing or unreadable means every command that
    reads `os.getenv(...)` will silently fall back to defaults — which
    includes the startup validators that gate share-auth."""
    env_path = project_root / ".env"
    if not env_path.exists():
        return CheckResult(
            name="env_file",
            status=Status.WARN,
            message=(
                f".env not found at {env_path}. The backend will boot with "
                "default env values; share-auth will be off, no provider "
                "tokens, etc. OK for a smoke test, not for a real deploy."
            ),
            details={"path": str(env_path)},
        )
    if not os.access(env_path, os.R_OK):
        return CheckResult(
            name="env_file",
            status=Status.FAIL,
            message=f".env exists at {env_path} but is not readable.",
            details={"path": str(env_path)},
        )
    return CheckResult(
        name="env_file",
        status=Status.PASS,
        message=f".env present and readable ({env_path.stat().st_size} bytes).",
        details={"path": str(env_path)},
    )


def check_python_venv(project_root: Path = _PROJECT_ROOT) -> CheckResult:
    """Every launchd wrapper hardcodes ``${PROJECT_ROOT}/.venv311/bin/python``.
    If that's not present, all five jobs will fail with
    ``No such file or directory`` on first fire."""
    py = project_root / ".venv311" / "bin" / "python"
    if not py.exists():
        return CheckResult(
            name="python_venv",
            status=Status.FAIL,
            message=(
                f".venv311 python not found at {py}. Every launchd wrapper "
                f"hardcodes this path; jobs will fail on first fire. Run "
                f"`uv venv .venv311 --python 3.11` from {project_root}."
            ),
            details={"path": str(py)},
        )
    if not os.access(py, os.X_OK):
        return CheckResult(
            name="python_venv",
            status=Status.FAIL,
            message=f".venv311 python exists at {py} but is not executable.",
            details={"path": str(py)},
        )
    return CheckResult(
        name="python_venv",
        status=Status.PASS,
        message=f".venv311 python present and executable.",
        details={"path": str(py)},
    )


def check_frontend_dist(project_root: Path = _PROJECT_ROOT) -> CheckResult:
    """Production deployments serve the React app same-origin from
    ``web/frontend/dist/``. Missing is a WARN, not a FAIL — dev mode
    (Vite on :5173) works fine without it. The runbook is explicit
    that you need ``npm run build`` before flipping share-auth on for
    a real deploy."""
    dist = project_root / "web" / "frontend" / "dist"
    index = dist / "index.html"
    if not index.exists():
        return CheckResult(
            name="frontend_dist",
            status=Status.WARN,
            message=(
                f"Frontend not built at {dist}. Dev mode works without "
                "this, but production same-origin deployment needs "
                "`cd web/frontend && npm run build` first."
            ),
            details={"path": str(dist)},
        )
    return CheckResult(
        name="frontend_dist",
        status=Status.PASS,
        message=f"Frontend build present ({dist}).",
        details={"path": str(dist)},
    )


def check_wrapper_scripts(project_root: Path = _PROJECT_ROOT) -> CheckResult:
    """All five launchd wrappers must exist and be executable. The
    install script copies their absolute paths into the rendered
    plists, so a missing or non-executable wrapper means launchd will
    silently log permission errors."""
    auto = project_root / "scripts" / "automation"
    missing: list[str] = []
    not_exec: list[str] = []
    for name in _WRAPPER_NAMES:
        p = auto / name
        if not p.exists():
            missing.append(name)
        elif not os.access(p, os.X_OK):
            not_exec.append(name)
    if missing:
        return CheckResult(
            name="wrapper_scripts",
            status=Status.FAIL,
            message=f"Missing wrapper scripts: {missing}",
            details={"missing": missing, "not_executable": not_exec},
        )
    if not_exec:
        return CheckResult(
            name="wrapper_scripts",
            status=Status.FAIL,
            message=f"Wrapper scripts present but not executable: {not_exec}",
            details={"missing": missing, "not_executable": not_exec},
        )
    return CheckResult(
        name="wrapper_scripts",
        status=Status.PASS,
        message=f"All {len(_WRAPPER_NAMES)} launchd wrappers present and executable.",
        details={"count": len(_WRAPPER_NAMES)},
    )


def check_plist_templates(project_root: Path = _PROJECT_ROOT) -> CheckResult:
    """All five plists exist and contain the placeholders the install
    script substitutes (``__PROJECT_ROOT__`` and ``__HOME__``).
    Detects the case where someone hand-edited a plist and removed
    the placeholders — the install script would then leave the wrong
    paths in the rendered output, but everything LOOKS fine.
    """
    auto = project_root / "scripts" / "automation"
    issues: dict[str, str] = {}
    for name in _PLIST_NAMES:
        p = auto / name
        if not p.exists():
            issues[name] = "missing"
            continue
        text = p.read_text(encoding="utf-8")
        if "__PROJECT_ROOT__" not in text:
            issues[name] = "missing __PROJECT_ROOT__ placeholder"
        elif "__HOME__" not in text:
            issues[name] = "missing __HOME__ placeholder"
    if issues:
        return CheckResult(
            name="plist_templates",
            status=Status.FAIL,
            message=f"Plist template issues: {issues}",
            details={"issues": issues},
        )
    return CheckResult(
        name="plist_templates",
        status=Status.PASS,
        message=f"All {len(_PLIST_NAMES)} plists present with placeholders intact.",
        details={"count": len(_PLIST_NAMES)},
    )


def check_launchagents_installed(
    launch_agents_dir: Path = _LAUNCH_AGENTS,
) -> CheckResult:
    """How many of the 5 plists are installed to ``~/Library/LaunchAgents``.

    Zero installed → SKIP (likely a fresh checkout or dev box). Partial
    → WARN (operator probably forgot to re-run installer after a
    PR-added job — that was the case for the log-rotation job after
    PR #61). All five → PASS.
    """
    if not launch_agents_dir.exists():
        return CheckResult(
            name="launchagents_installed",
            status=Status.SKIP,
            message=(
                f"{launch_agents_dir} does not exist (no LaunchAgents on "
                "this machine). Run scripts/automation/install_launchd_jobs.sh "
                "to install."
            ),
            details={"installed_count": 0, "total": len(_PLIST_NAMES)},
        )
    installed = [
        name for name in _PLIST_NAMES if (launch_agents_dir / name).exists()
    ]
    if not installed:
        return CheckResult(
            name="launchagents_installed",
            status=Status.SKIP,
            message=(
                f"No LaunchAgents installed. Run "
                "scripts/automation/install_launchd_jobs.sh when ready."
            ),
            details={"installed_count": 0, "total": len(_PLIST_NAMES)},
        )
    if len(installed) < len(_PLIST_NAMES):
        missing = sorted(set(_PLIST_NAMES) - set(installed))
        return CheckResult(
            name="launchagents_installed",
            status=Status.WARN,
            message=(
                f"Partial install: {len(installed)} of {len(_PLIST_NAMES)} "
                f"LaunchAgents present. Re-run install_launchd_jobs.sh to "
                f"pick up: {missing}"
            ),
            details={
                "installed_count": len(installed),
                "total": len(_PLIST_NAMES),
                "missing": missing,
            },
        )
    return CheckResult(
        name="launchagents_installed",
        status=Status.PASS,
        message=f"All {len(_PLIST_NAMES)} LaunchAgents installed.",
        details={"installed_count": len(installed), "total": len(_PLIST_NAMES)},
    )


def check_startup_validators(
    project_root: Path = _PROJECT_ROOT,
    python_bin: Path | None = None,
    timeout_seconds: float = 30.0,
) -> CheckResult:
    """Import ``web.api.app`` in a subprocess to exercise the PR #59
    startup validators. If the import succeeds, the current .env is
    valid for the current SHARE_AUTH posture.

    Subprocess isolation is deliberate: the import has side effects
    (loads .env, registers FastAPI routes, opens log handles). Running
    it in the parent process would pollute state and make the preflight
    non-reentrant.
    """
    py = python_bin or (project_root / ".venv311" / "bin" / "python")
    if not py.exists():
        return CheckResult(
            name="startup_validators",
            status=Status.SKIP,
            message=(
                f"Cannot run startup validators: {py} not found. "
                "(Earlier python_venv check should have already flagged this.)"
            ),
            details={"python_bin": str(py)},
        )
    # ``-c "import web.api.app"`` triggers _validate_auth_config() at
    # module load. Print a known marker so we can distinguish "import
    # succeeded silently" from "stdout was empty due to early exit".
    try:
        result = subprocess.run(
            [str(py), "-c", "import web.api.app; print('preflight_ok')"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="startup_validators",
            status=Status.FAIL,
            message=(
                f"web.api.app import timed out after {timeout_seconds}s "
                "(infinite loop at module load or pathologically slow .env)."
            ),
            details={"timeout_seconds": timeout_seconds},
        )
    if result.returncode == 0 and "preflight_ok" in result.stdout:
        return CheckResult(
            name="startup_validators",
            status=Status.PASS,
            message="web.api.app loads cleanly; PR #59 startup validators all pass.",
            details={},
        )
    # The validator's RuntimeError messages are clear; surface the
    # last error line verbatim so the operator can copy-paste the env
    # var to fix.
    last_err_line = ""
    if result.stderr:
        for line in reversed(result.stderr.strip().splitlines()):
            if line.strip():
                last_err_line = line.strip()
                break
    return CheckResult(
        name="startup_validators",
        status=Status.FAIL,
        message=(
            f"web.api.app failed to import (rc={result.returncode}). "
            f"Last stderr line: {last_err_line or '(empty)'}"
        ),
        details={
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        },
    )


def check_sqlite_stores(stores: tuple[Path, ...] = _KNOWN_SQLITE_STORES) -> CheckResult:
    """``PRAGMA quick_check`` on each known SQLite path. Fresh installs
    have no stores yet → SKIP. Existing stores must report "ok"; any
    other result → FAIL (corruption requires manual recovery).
    """
    checked: dict[str, str] = {}
    bad: dict[str, str] = {}
    for path in stores:
        if not path.exists():
            checked[str(path)] = "absent"
            continue
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                row = conn.execute("PRAGMA quick_check;").fetchone()
            verdict = row[0] if row else "(no row)"
            checked[str(path)] = verdict
            if verdict != "ok":
                bad[str(path)] = verdict
        except sqlite3.Error as exc:
            bad[str(path)] = f"sqlite3.Error: {exc}"
            checked[str(path)] = "error"
    if bad:
        return CheckResult(
            name="sqlite_stores",
            status=Status.FAIL,
            message=f"SQLite quick_check failed for: {sorted(bad)}",
            details={"checked": checked, "failures": bad},
        )
    if all(v == "absent" for v in checked.values()):
        return CheckResult(
            name="sqlite_stores",
            status=Status.SKIP,
            message="No SQLite stores yet — fresh install.",
            details={"checked": checked},
        )
    return CheckResult(
        name="sqlite_stores",
        status=Status.PASS,
        message=(
            f"{sum(1 for v in checked.values() if v == 'ok')} SQLite store(s) "
            f"pass quick_check; {sum(1 for v in checked.values() if v == 'absent')} "
            "absent."
        ),
        details={"checked": checked},
    )


def check_log_rotation_dryrun(
    project_root: Path = _PROJECT_ROOT,
    python_bin: Path | None = None,
    timeout_seconds: float = 30.0,
) -> CheckResult:
    """Dry-run the log rotator. It walks ``~/.options_calculator_pro/logs/``
    and reports what it would rotate — no mutations. Exit 0 means
    the rotator's discovery + size logic is intact.
    """
    py = python_bin or (project_root / ".venv311" / "bin" / "python")
    if not py.exists():
        return CheckResult(
            name="log_rotation_dryrun",
            status=Status.SKIP,
            message=f"Cannot run rotator dry-run: {py} not found.",
            details={},
        )
    try:
        result = subprocess.run(
            [str(py), str(project_root / "scripts" / "rotate_launchd_logs.py"), "--dry-run"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="log_rotation_dryrun",
            status=Status.FAIL,
            message=f"rotate_launchd_logs.py --dry-run timed out after {timeout_seconds}s.",
            details={"timeout_seconds": timeout_seconds},
        )
    if result.returncode != 0:
        return CheckResult(
            name="log_rotation_dryrun",
            status=Status.FAIL,
            message=f"rotate_launchd_logs.py --dry-run exited with {result.returncode}.",
            details={
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            },
        )
    # Count what it found so the operator sees the rotator is wired.
    line_count = sum(1 for line in result.stdout.splitlines() if line.strip().startswith("["))
    return CheckResult(
        name="log_rotation_dryrun",
        status=Status.PASS,
        message=f"Rotator dry-run clean; {line_count} log file(s) scanned.",
        details={"files_scanned": line_count},
    )


def check_backend_health(
    base_url: str = "http://127.0.0.1:8000",
    timeout_seconds: float = 2.0,
) -> CheckResult:
    """Optional probe of ``/api/health``. SKIP if no backend is running
    (connection refused / DNS failure) — preflight is a deploy-time
    check, not a runtime monitor; we never start uvicorn here. PASS
    if 200 OK with a recognizable body; FAIL on any other response."""
    url = f"{base_url.rstrip('/')}/api/health"
    try:
        with urlopen(Request(url), timeout=timeout_seconds) as resp:  # noqa: S310 — local URL
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status == 200:
                # Light schema check: backend returns {"status": "ok", ...}.
                try:
                    payload = json.loads(body)
                    ok = payload.get("status") == "ok"
                except json.JSONDecodeError:
                    ok = False
                if ok:
                    return CheckResult(
                        name="backend_health",
                        status=Status.PASS,
                        message=f"Backend responding 200 OK at {url}.",
                        details={"url": url},
                    )
                return CheckResult(
                    name="backend_health",
                    status=Status.FAIL,
                    message=(
                        f"Backend at {url} returned 200 but body wasn't the "
                        "expected health shape."
                    ),
                    details={"url": url, "body_head": body[:200]},
                )
            return CheckResult(
                name="backend_health",
                status=Status.FAIL,
                message=f"Backend at {url} returned HTTP {resp.status}.",
                details={"url": url, "status_code": resp.status},
            )
    except HTTPError as exc:
        # Codex P1 (PR #63 follow-up): HTTPError is a subclass of
        # URLError. Without this explicit branch, a backend that's
        # UP but returning 4xx/5xx (auth misconfig → 401, app crash
        # → 500) would have been swallowed by the URLError block
        # below as "no backend reachable / SKIP" — green-lighting a
        # broken deployment. A reachable backend that refuses must
        # FAIL, not SKIP.
        return CheckResult(
            name="backend_health",
            status=Status.FAIL,
            message=(
                f"Backend at {url} is up but returned HTTP {exc.code} "
                f"({exc.reason})."
            ),
            details={"url": url, "status_code": exc.code, "reason": str(exc.reason)},
        )
    except URLError as exc:
        # Connection refused / DNS fail / TLS handshake error → SKIP.
        # Preflight isn't responsible for *starting* the backend;
        # it's checking that IF you have one running, it's healthy.
        return CheckResult(
            name="backend_health",
            status=Status.SKIP,
            message=(
                f"No backend reachable at {url} ({type(exc).__name__}: "
                f"{exc.reason if hasattr(exc, 'reason') else exc}). Skipped — "
                "preflight does not start uvicorn."
            ),
            details={"url": url},
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="backend_health",
            status=Status.FAIL,
            message=f"Unexpected error probing {url}: {type(exc).__name__}: {exc}",
            details={"url": url},
        )


# ── Aggregation + output ─────────────────────────────────────────────────


_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_env_file,
    check_python_venv,
    check_frontend_dist,
    check_wrapper_scripts,
    check_plist_templates,
    check_launchagents_installed,
    check_startup_validators,
    check_sqlite_stores,
    check_log_rotation_dryrun,
    check_backend_health,
)


def run_all_checks() -> list[CheckResult]:
    """Run every check in declaration order. Each check is independent;
    a FAIL in one does NOT skip subsequent checks (we want the
    operator to see all the issues at once, not play whack-a-mole)."""
    return [fn() for fn in _CHECKS]


def aggregate_exit_code(results: list[CheckResult]) -> int:
    """Watchdog-convention exit codes:
      0 — all PASS or SKIP
      1 — at least one WARN, no FAIL
      2 — at least one FAIL
    """
    if any(r.status == Status.FAIL for r in results):
        return 2
    if any(r.status == Status.WARN for r in results):
        return 1
    return 0


_STATUS_GLYPHS = {
    Status.PASS: "✓",
    Status.WARN: "~",
    Status.FAIL: "✗",
    Status.SKIP: "·",
}


def format_human(results: list[CheckResult]) -> str:
    """Human-readable summary. Cap message length so terminals don't
    wrap awkwardly, but keep the full message in --json output."""
    out: list[str] = ["Pre-deployment smoke check", ""]
    width = max(len(r.name) for r in results)
    for r in results:
        glyph = _STATUS_GLYPHS[r.status]
        msg = r.message
        if len(msg) > 100:
            msg = msg[:97] + "..."
        out.append(f"  {glyph} {r.status.value:<4} {r.name:<{width}}  {msg}")
    counts = {s: sum(1 for r in results if r.status == s) for s in Status}
    out.append("")
    out.append(
        f"  Summary: {counts[Status.PASS]} PASS, {counts[Status.WARN]} WARN, "
        f"{counts[Status.FAIL]} FAIL, {counts[Status.SKIP]} SKIP"
    )
    return "\n".join(out)


def format_json(results: list[CheckResult]) -> str:
    return json.dumps(
        {
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                }
                for r in results
            ],
            "summary": {
                s.value: sum(1 for r in results if r.status == s) for s in Status
            },
        },
        indent=2,
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-deployment / pre-start smoke check for options-calculator-pro.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON (for CI ingestion). Default is human-readable.",
    )
    args = parser.parse_args(argv)
    results = run_all_checks()
    if args.json:
        print(format_json(results))
    else:
        print(format_human(results))
    return aggregate_exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
