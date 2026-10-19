"""Which source generation produced an artifact, and is it the current one.

An archive, an evidence report or a served frontend that cannot name the
commit it came from cannot be tied to the code that wrote it. Every helper
here degrades visibly: an unknown identity is reported as unknown (``None``
plus an ``error``), never guessed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from services import external_io_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "origin"
DEFAULT_BRANCH = "main"


def _git(repo_root: Path, *args: str, timeout: float) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or f"git {args[0]} exited {result.returncode}")
    return result.stdout.strip()


def source_identity(repo_root: Path = REPO_ROOT, *, timeout: float = 10.0) -> dict[str, Any]:
    """The checkout's commit, branch and whether tracked files differ from it."""
    try:
        sha = _git(repo_root, "rev-parse", "HEAD", timeout=timeout)
        branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD", timeout=timeout)
        dirty = bool(_git(repo_root, "status", "--porcelain", "--untracked-files=no", timeout=timeout))
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"git_sha": None, "branch": None, "dirty": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"git_sha": sha, "branch": branch, "dirty": dirty, "error": None}


def deployment_drift(
    repo_root: Path = REPO_ROOT,
    *,
    remote: str = DEFAULT_REMOTE,
    branch: str = DEFAULT_BRANCH,
    fetch: bool = True,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """How far the checkout is from ``remote/branch``.

    The fetch is best effort: when it fails the comparison uses the last
    fetched ref and says so, so a network outage cannot pass for "current".
    """
    identity = source_identity(repo_root)
    upstream = f"{remote}/{branch}"
    out: dict[str, Any] = {**identity, "upstream": upstream, "upstream_sha": None,
                           "behind": None, "ahead": None, "fetched": False, "fetch_error": None}
    if identity["git_sha"] is None:
        return out
    if fetch:
        try:
            external_io_gate.assert_allowed(external_io_gate.Category.GIT_REMOTE)
            _git(repo_root, "fetch", "--quiet", remote, branch, timeout=timeout)
            out["fetched"] = True
        except (external_io_gate.ExternalIOBlocked, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            out["fetch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["upstream_sha"] = _git(repo_root, "rev-parse", "--verify", f"{upstream}^{{commit}}", timeout=10.0)
        ahead, behind = _git(repo_root, "rev-list", "--left-right", "--count",
                             f"HEAD...{upstream}", timeout=10.0).split()
        out["ahead"], out["behind"] = int(ahead), int(behind)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _short(sha: Any) -> str:
    return str(sha)[:7] if sha else "unknown"


def build_code_identity_health(
    drift: Mapping[str, Any],
    report_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Watchdog issues for code drift, in the evidence_health issue shape.

    ``drift`` is :func:`deployment_drift` for the checkout the jobs run from;
    ``report_identity`` is the ``code_identity`` the evidence cycle stamped
    into today's report (``None`` when the report predates the stamp).
    """
    issues: list[dict[str, Any]] = []
    head = drift.get("git_sha")
    if head is None:
        issues.append({"severity": "WARN", "alertable": False,
                       "message": f"Cannot identify the running code ({drift.get('error') or 'not a git checkout'})."})
    else:
        behind = drift.get("behind")
        if behind is None:
            issues.append({"severity": "WARN", "alertable": False,
                           "message": f"Cannot compare the running code with {drift.get('upstream')} "
                                      f"({drift.get('error') or 'no upstream ref'})."})
        elif behind > 0:
            issues.append({"severity": "FAIL", "alertable": True,
                           "message": f"Running code {_short(head)} is {behind} commit(s) behind "
                                      f"{drift.get('upstream')} {_short(drift.get('upstream_sha'))}: "
                                      "pull, then restart the launchd jobs."})
        if drift.get("fetch_error"):
            issues.append({"severity": "WARN", "alertable": False,
                           "message": f"Could not fetch {drift.get('upstream')}; compared against the last "
                                      f"fetched ref ({drift.get('fetch_error')})."})
        if drift.get("dirty"):
            issues.append({"severity": "WARN", "alertable": False,
                           "message": f"Running checkout {_short(head)} has uncommitted changes to tracked files."})
    if report_identity is None:
        issues.append({"severity": "WARN", "alertable": False,
                       "message": "Today's evidence report does not record which code produced it."})
    else:
        report_sha = report_identity.get("git_sha")
        if report_sha is None:
            issues.append({"severity": "WARN", "alertable": False,
                           "message": "The evidence cycle could not identify the code it ran "
                                      f"({report_identity.get('error') or 'unknown'})."})
        elif head is not None and report_sha != head:
            issues.append({"severity": "WARN", "alertable": True,
                           "message": f"The evidence cycle ran code {_short(report_sha)} but this checkout is "
                                      f"{_short(head)}: two checkouts are writing state, or the code changed "
                                      "since the cycle ran."})
    return {
        "issues": issues,
        "summary": {
            "git_sha": head,
            "dirty": drift.get("dirty"),
            "upstream": drift.get("upstream"),
            "upstream_sha": drift.get("upstream_sha"),
            "behind": drift.get("behind"),
            "ahead": drift.get("ahead"),
            "fetched": drift.get("fetched"),
            "report_git_sha": (report_identity or {}).get("git_sha"),
        },
    }
