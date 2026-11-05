"""The IB Gateway launchd job: the plist keeps the gateway running, the wrapper
finds the app and never starts a second instance, and the installer, the
uninstaller and the README all know the job.
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
PLIST = AUTOMATION / "com.optionscalculator.ib-gateway.plist"
WRAPPER = AUTOMATION / "run_ib_gateway.sh"


def test_plist_keeps_the_gateway_running_in_the_gui_session() -> None:
    data = plistlib.loads(PLIST.read_bytes())
    assert data["Label"] == "com.optionscalculator.ib-gateway"
    assert data["ProgramArguments"] == ["__PROJECT_ROOT__/scripts/automation/run_ib_gateway.sh"]
    assert data["RunAtLoad"] is True
    # Restart on failure only: Quit in the gateway or `launchctl stop` must stick.
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ThrottleInterval"] >= 30
    # A GUI app: never started in a headless session.
    assert data["LimitLoadToSessionType"] == "Aqua"
    assert data["ProcessType"] == "Interactive"
    raw = PLIST.read_text()
    for leak in ("/Users/", "/home/"):
        assert leak not in raw


def test_plist_renders_like_the_other_jobs() -> None:
    rendered = PLIST.read_text().replace("__PROJECT_ROOT__", "/p").replace("__HOME__", "/h")
    data = plistlib.loads(rendered.encode())
    assert data["ProgramArguments"] == ["/p/scripts/automation/run_ib_gateway.sh"]
    assert data["StandardOutPath"] == "/h/.options_calculator_pro/logs/ib_gateway_stdout.log"
    assert data["StandardErrorPath"] == "/h/.options_calculator_pro/logs/ib_gateway_stderr.log"


def test_gateway_logs_are_outside_the_rotation_that_would_orphan_them() -> None:
    data = plistlib.loads(PLIST.read_bytes())
    for key in ("StandardOutPath", "StandardErrorPath"):
        name = Path(data[key]).name
        assert not any(fnmatch.fnmatch(name, g) for g in _LAUNCHD_LOG_GLOBS), name


def test_installer_uninstaller_and_readme_know_the_job() -> None:
    for script in ("install_launchd_jobs.sh", "uninstall_launchd_jobs.sh"):
        assert "com.optionscalculator.ib-gateway.plist" in (AUTOMATION / script).read_text(), script
    body = (AUTOMATION / "README.md").read_text()
    assert "`com.optionscalculator.ib-gateway`" in body
    assert "does not sign in" in body          # the limit is stated, not implied away
    assert "localhost only" in body            # how to close the network-visible port


def test_wrapper_is_executable() -> None:
    assert os.access(WRAPPER, os.X_OK)


# ── the wrapper, run for real against a fake app ────────────────────────────


def _fake_project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "proj"
    (root / "scripts" / "automation").mkdir(parents=True)
    shutil.copy2(WRAPPER, root / "scripts" / "automation" / WRAPPER.name)
    app = tmp_path / "IB Gateway 10.51.app"
    stub = app / "Contents" / "MacOS" / "JavaApplicationStub"
    stub.parent.mkdir(parents=True)
    # Records that it was exec'd (and from where), then exits cleanly.
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "stub-ran" "$PWD" > "$CALLS"\nexit 0\n')
    stub.chmod(0o755)
    return root, app


def _run(root: Path, tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    calls = tmp_path / "calls.txt"
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    base = {k: v for k, v in os.environ.items() if not k.startswith("IB_GATEWAY_")}
    base.update({"HOME": str(home), "CALLS": str(calls), **env})
    result = subprocess.run(["bash", str(root / "scripts" / "automation" / WRAPPER.name)],
                            env=base, capture_output=True, text=True, timeout=60)
    result.calls = calls.read_text().splitlines() if calls.exists() else []  # type: ignore[attr-defined]
    return result


def test_wrapper_execs_the_app_named_in_the_env_file(tmp_path: Path) -> None:
    root, app = _fake_project(tmp_path)
    (root / ".env").write_text(f'IB_GATEWAY_APP="{app}"\n')
    result = _run(root, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.calls[:1] == ["stub-ran"]
    assert result.calls[1] == str(root)             # cwd is the project root
    assert f"ib gateway start app={app}" in result.stdout
    assert (tmp_path / "home" / ".options_calculator_pro" / "logs").is_dir()


def test_process_env_overrides_the_env_file(tmp_path: Path) -> None:
    root, app = _fake_project(tmp_path)
    (root / ".env").write_text("IB_GATEWAY_APP=/nowhere/IB Gateway 1.0.app\n")
    result = _run(root, tmp_path, IB_GATEWAY_APP=str(app))
    assert result.returncode == 0, result.stderr
    assert result.calls[:1] == ["stub-ran"]


def test_wrapper_finds_the_newest_gateway_under_applications(tmp_path: Path) -> None:
    root, app = _fake_project(tmp_path)
    home = tmp_path / "home"
    (home / "Applications").mkdir(parents=True)
    old = home / "Applications" / "IB Gateway 10.9.app"
    shutil.copytree(app, old)
    (old / "Contents" / "MacOS" / "JavaApplicationStub").write_text(
        '#!/bin/sh\nprintf "%s\\n" "old-stub-ran" > "$CALLS"\n')
    shutil.copytree(app, home / "Applications" / "IB Gateway 10.51.app")
    result = _run(root, tmp_path)   # no IB_GATEWAY_APP anywhere
    assert result.returncode == 0, result.stderr
    assert result.calls[:1] == ["stub-ran"]       # 10.51 beats 10.9 under version sort
    assert "IB Gateway 10.51.app" in result.stdout


def test_wrapper_finds_the_installer_layout_with_a_version_folder(tmp_path: Path) -> None:
    # IB's installer: ~/Applications/IB Gateway 10.51/IB Gateway 10.51.app
    root, app = _fake_project(tmp_path)
    home = tmp_path / "home"
    nested = home / "Applications" / "IB Gateway 10.51" / "IB Gateway 10.51.app"
    nested.parent.mkdir(parents=True)
    shutil.copytree(app, nested)
    # The installer also leaves an uninstaller beside it, which sorts LAST under
    # version sort and once got launched instead of the gateway. Never again.
    uninstaller = nested.parent / "IB Gateway 10.51 Uninstaller.app"
    shutil.copytree(app, uninstaller)
    (uninstaller / "Contents" / "MacOS" / "JavaApplicationStub").write_text(
        '#!/bin/sh\nprintf "%s\\n" "UNINSTALLER-RAN" > "$CALLS"\n')
    result = _run(root, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.calls[:1] == ["stub-ran"]
    assert str(nested) in result.stdout
    assert "Uninstaller" not in result.stdout


def test_wrapper_without_an_app_exits_with_config_error(tmp_path: Path) -> None:
    root, _ = _fake_project(tmp_path)
    result = _run(root, tmp_path, IB_GATEWAY_APP=str(tmp_path / "missing.app"))
    assert result.returncode == 78
    assert "No IB Gateway app" in result.stderr and "IB_GATEWAY_APP" in result.stderr
    assert result.calls == []


def test_wrapper_never_sources_the_env_file(tmp_path: Path) -> None:
    root, app = _fake_project(tmp_path)
    marker = tmp_path / "sourced"
    (root / ".env").write_text(f'IB_GATEWAY_APP="{app}"\n$(touch {marker})\n`touch {marker}`\n')
    result = _run(root, tmp_path)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
