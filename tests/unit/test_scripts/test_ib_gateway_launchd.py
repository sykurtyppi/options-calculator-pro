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
import time
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
    # IB's daily auto-logoff can exit cleanly, so production supervision must
    # restart the gateway after both successful and failed exits. Operators use
    # launchctl bootout/unload when they intentionally need it stopped.
    assert data["KeepAlive"] is True
    assert data["ThrottleInterval"] >= 30
    # IB's auto-restart relaunches the gateway as a child of the old process;
    # without this launchd kills that relaunched, signed-in gateway when the
    # old one exits.
    assert data["AbandonProcessGroup"] is True
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
    # Records that it was exec'd (and from where), then exits cleanly. With
    # FAKE_GATEWAY_SECONDS set it instead plays a running gateway: it stays
    # alive that long and records nothing.
    stub.write_text(
        '#!/bin/sh\n'
        'if [ -n "$FAKE_GATEWAY_SECONDS" ]; then sleep "$FAKE_GATEWAY_SECONDS"; exit 0; fi\n'
        'printf "%s\\n" "stub-ran" "$PWD" > "$CALLS"\nexit 0\n')
    stub.chmod(0o755)
    return root, app


def _run(root: Path, tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    calls = tmp_path / "calls.txt"
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    base = {k: v for k, v in os.environ.items() if not k.startswith(("IB_GATEWAY_", "FAKE_GATEWAY_"))}
    base.update({"HOME": str(home), "CALLS": str(calls),
                 "IB_GATEWAY_ADOPT_GRACE_SECONDS": "0", "IB_GATEWAY_POLL_SECONDS": "1", **env})
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


# ── adopting a gateway that is already running ──────────────────────────────


def _running_gateway(app: Path, seconds: int, delay: float = 0.0) -> subprocess.Popen:
    """Start the fake stub as a long-running gateway (optionally after a delay),
    the way IB's auto-restart relaunches the real one: same launcher path."""
    stub = app / "Contents" / "MacOS" / "JavaApplicationStub"
    env = {**os.environ, "FAKE_GATEWAY_SECONDS": str(seconds)}
    if delay:
        return subprocess.Popen(["/bin/sh", "-c", f'sleep {delay}; exec "$0"', str(stub)], env=env)
    return subprocess.Popen([str(stub)], env=env)


def test_wrapper_adopts_a_running_gateway_instead_of_starting_a_second(tmp_path: Path) -> None:
    root, app = _fake_project(tmp_path)
    gateway = _running_gateway(app, seconds=3)
    try:
        time.sleep(0.3)
        result = _run(root, tmp_path, IB_GATEWAY_APP=str(app))
    finally:
        gateway.wait(10)
    assert result.returncode == 75                   # hands back to launchd once it is gone
    assert result.calls == []                        # the stub was never exec'd a second time
    assert "already running" in result.stdout and "adopting it" in result.stdout
    assert "adopted ib gateway exited" in result.stdout
    assert "ib gateway start app=" not in result.stdout


def test_wrapper_waits_out_the_grace_for_a_gateway_relaunching_itself(tmp_path: Path) -> None:
    # IB's auto-restart: the old process is gone (launchd reran the wrapper)
    # and the relaunched gateway shows up a moment later. The wrapper must
    # find it within the grace period, not start a fresh one at a login window.
    root, app = _fake_project(tmp_path)
    gateway = _running_gateway(app, seconds=3, delay=1.5)
    try:
        result = _run(root, tmp_path, IB_GATEWAY_APP=str(app), IB_GATEWAY_ADOPT_GRACE_SECONDS="8")
    finally:
        gateway.wait(15)
    assert result.returncode == 75
    assert result.calls == []
    assert "adopting it" in result.stdout


def test_wrapper_starts_a_gateway_when_the_grace_passes_with_none_running(tmp_path: Path) -> None:
    root, app = _fake_project(tmp_path)
    started = time.monotonic()
    result = _run(root, tmp_path, IB_GATEWAY_APP=str(app), IB_GATEWAY_ADOPT_GRACE_SECONDS="2")
    assert result.returncode == 0, result.stderr
    assert result.calls[:1] == ["stub-ran"]
    assert time.monotonic() - started >= 2          # it did wait before starting


def test_wrapper_reads_the_grace_from_the_env_file_and_ignores_garbage(tmp_path: Path) -> None:
    root, app = _fake_project(tmp_path)
    (root / ".env").write_text(f'IB_GATEWAY_APP="{app}"\nIB_GATEWAY_ADOPT_GRACE_SECONDS=2\n')
    base = {k: v for k, v in os.environ.items() if not k.startswith(("IB_GATEWAY_", "FAKE_GATEWAY_"))}
    home = tmp_path / "home"; home.mkdir(exist_ok=True)
    calls = tmp_path / "calls.txt"
    started = time.monotonic()
    done = subprocess.run(["bash", str(root / "scripts" / "automation" / WRAPPER.name)],
                          env={**base, "HOME": str(home), "CALLS": str(calls)},
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert 2 <= time.monotonic() - started < 15      # the .env value, not the 20s default
    # A non-numeric value falls back to the default instead of breaking the wrapper.
    result = _run(root, tmp_path, IB_GATEWAY_APP=str(app), IB_GATEWAY_POLL_SECONDS="soon")
    assert result.returncode == 0, result.stderr
