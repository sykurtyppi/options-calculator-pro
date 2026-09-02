"""Codex audit F4: preflight must derive the launchd job list from the
installer instead of a hand-maintained copy that drifted to 5 of 8 and
certified a partial install as complete."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
AUTOMATION = REPO / "scripts" / "automation"


def _from_script(name: str) -> set[str]:
    text = (AUTOMATION / name).read_text()
    block = re.search(r"for\s+plist\s+in(.*?)\n\s*do\b", text, re.S).group(1)
    return set(re.findall(r"com\.optionscalculator\.[\w.-]+\.plist", block))


def test_preflight_job_list_matches_installer_uninstaller_and_templates():
    from scripts.preflight_check import installer_plist_names, wrapper_names_for

    derived = set(installer_plist_names(REPO))
    assert derived == _from_script("install_launchd_jobs.sh")
    assert derived == _from_script("uninstall_launchd_jobs.sh"), "install/uninstall lists drifted"
    on_disk = {p.name for p in AUTOMATION.glob("com.optionscalculator.*.plist")}
    assert derived == on_disk, f"installer vs templates on disk: {derived ^ on_disk}"
    assert len(derived) >= 8
    wrappers = wrapper_names_for(tuple(sorted(derived)), REPO)
    assert len(wrappers) == len(derived)
    assert all((AUTOMATION / w).exists() for w in wrappers)


def test_partial_install_is_not_certified_complete(tmp_path: Path):
    """Reproduces the false PASS: only the 5 legacy jobs installed."""
    from scripts.preflight_check import (
        Status,
        check_launchagents_installed,
        installer_plist_names,
    )

    legacy = {
        "com.optionscalculator.candidate-exit-resolver.plist",
        "com.optionscalculator.evidence-cycle.plist",
        "com.optionscalculator.evidence-watchdog.plist",
        "com.optionscalculator.weekly-evidence-report.plist",
        "com.optionscalculator.log-rotation.plist",
    }
    agents = tmp_path / "LaunchAgents"; agents.mkdir()
    for name in legacy:
        (agents / name).write_text("<plist/>")
    result = check_launchagents_installed(agents, project_root=REPO)
    assert result.status is Status.WARN, result
    assert result.details["total"] == len(installer_plist_names(REPO))
    assert result.details["installed_count"] == 5
    assert "premarket-screener-alert" in " ".join(result.details["missing"])


def test_preflight_refuses_to_certify_without_a_job_list(tmp_path: Path):
    import pytest as _pytest

    from scripts.preflight_check import installer_plist_names

    fake_root = tmp_path / "root"
    (fake_root / "scripts" / "automation").mkdir(parents=True)
    (fake_root / "scripts" / "automation" / "install_launchd_jobs.sh").write_text("#!/bin/bash\necho nothing\n")
    with _pytest.raises(RuntimeError):
        installer_plist_names(fake_root)
