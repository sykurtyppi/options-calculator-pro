"""Code identity: backup manifests, report stamps, drift alerts, /api/version."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import tarfile
from datetime import date
from pathlib import Path

import pytest

from scripts import backup_state, restore_state
from scripts import run_evidence_cycle as evidence_cycle
from scripts import watch_daily_evidence_cycle as watch
from services import external_io_gate
from services.code_identity import build_code_identity_health, deployment_drift, source_identity

SHA_A = "a" * 40
SHA_B = "b" * 40


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "f.txt").write_text("1\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "one")
    return repo


# ── source identity / drift ────────────────────────────────────────────────


def test_source_identity_names_the_commit_and_dirty_tracked_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    assert source_identity(repo) == {"git_sha": head, "branch": "main", "dirty": False, "error": None}
    (repo / "untracked.txt").write_text("x")
    assert source_identity(repo)["dirty"] is False  # untracked files are not the code
    (repo / "f.txt").write_text("2\n")
    assert source_identity(repo)["dirty"] is True


def test_source_identity_outside_a_checkout_is_unknown_not_guessed(tmp_path: Path) -> None:
    out = source_identity(tmp_path)
    assert out["git_sha"] is None and out["dirty"] is None and out["error"]


def test_drift_counts_commits_behind_the_last_fetched_upstream(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    old = _git(repo, "rev-parse", "HEAD")
    (repo / "f.txt").write_text("2\n")
    _git(repo, "commit", "-q", "-am", "two")
    new = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", new)
    _git(repo, "checkout", "-q", old)

    out = deployment_drift(repo, fetch=False)
    assert (out["git_sha"], out["upstream_sha"], out["behind"], out["ahead"]) == (old, new, 1, 0)
    assert out["fetched"] is False and out["fetch_error"] is None


def test_drift_fetch_failure_is_reported_and_falls_back_to_the_last_fetched_ref(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    # Blocked by the test-session gate: never silently skipped.
    out = deployment_drift(repo, fetch=True)
    assert "ExternalIOBlocked" in out["fetch_error"] and out["fetched"] is False
    assert out["behind"] == 0
    # Allowed but no such remote: still a reported failure.
    external_io_gate.enable(external_io_gate.Category.GIT_REMOTE)
    try:
        out = deployment_drift(repo, fetch=True)
    finally:
        external_io_gate.disable(external_io_gate.Category.GIT_REMOTE)
    assert out["fetch_error"] and out["fetched"] is False and out["behind"] == 0


def test_drift_fetches_the_upstream_before_comparing(tmp_path: Path) -> None:
    remote = _repo(tmp_path)  # a local "origin": fetching it needs no network
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(remote), str(clone)], check=True, capture_output=True)
    (remote / "f.txt").write_text("2\n")
    _git(remote, "commit", "-q", "-am", "two")
    external_io_gate.enable(external_io_gate.Category.GIT_REMOTE)
    try:
        assert deployment_drift(clone, fetch=False)["behind"] == 0  # not fetched yet
        out = deployment_drift(clone, fetch=True)
    finally:
        external_io_gate.disable(external_io_gate.Category.GIT_REMOTE)
    assert out["fetched"] is True and out["fetch_error"] is None
    assert out["behind"] == 1 and out["upstream_sha"] == _git(remote, "rev-parse", "HEAD")


def test_drift_without_an_upstream_ref_is_unknown(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = deployment_drift(repo, fetch=False)
    assert out["behind"] is None and out["upstream_sha"] is None and out["error"]


# ── watchdog health ─────────────────────────────────────────────────────────


def _drift(**over):
    base = {"git_sha": SHA_A, "dirty": False, "error": None, "upstream": "origin/main",
            "upstream_sha": SHA_A, "behind": 0, "ahead": 0, "fetched": True, "fetch_error": None}
    return {**base, **over}


def _issues(drift, report):
    return [(i["severity"], i["alertable"], i["message"]) for i in build_code_identity_health(drift, report)["issues"]]


def test_current_clean_code_matching_the_report_has_no_issues() -> None:
    health = build_code_identity_health(_drift(), {"git_sha": SHA_A})
    assert health["issues"] == []
    assert health["summary"]["git_sha"] == SHA_A and health["summary"]["report_git_sha"] == SHA_A


def test_running_behind_upstream_is_an_alertable_failure() -> None:
    [(sev, alertable, msg)] = _issues(_drift(behind=17, upstream_sha=SHA_B), {"git_sha": SHA_A})
    assert (sev, alertable) == ("FAIL", True)
    assert "aaaaaaa is 17 commit(s) behind origin/main bbbbbbb" in msg


def test_one_commit_behind_already_alerts() -> None:
    [(sev, alertable, _msg)] = _issues(_drift(behind=1, upstream_sha=SHA_B), {"git_sha": SHA_A})
    assert (sev, alertable) == ("FAIL", True)


def test_report_from_other_code_is_alertable() -> None:
    [(sev, alertable, msg)] = _issues(_drift(), {"git_sha": SHA_B})
    assert (sev, alertable) == ("WARN", True)
    assert "ran code bbbbbbb but this checkout is aaaaaaa" in msg


@pytest.mark.parametrize("drift,report,needle", [
    (_drift(git_sha=None, error="not a repo"), {"git_sha": SHA_A}, "Cannot identify the running code (not a repo)"),
    (_drift(behind=None, error="no ref"), {"git_sha": SHA_A}, "Cannot compare the running code"),
    (_drift(fetch_error="offline"), {"git_sha": SHA_A}, "Could not fetch origin/main"),
    (_drift(dirty=True), {"git_sha": SHA_A}, "uncommitted changes"),
    (_drift(), None, "does not record which code produced it"),
    (_drift(), {"git_sha": None, "error": "boom"}, "could not identify the code it ran (boom)"),
])
def test_unknowns_are_visible_but_do_not_page(drift, report, needle) -> None:
    [(sev, alertable, msg)] = _issues(drift, report)
    assert (sev, alertable) == ("WARN", False)
    assert needle in msg


def test_watchdog_alerts_on_drift_and_keeps_unknowns_as_warnings() -> None:
    clean = {"ok": True, "errors": [], "warnings": []}
    code = build_code_identity_health(_drift(behind=3, fetch_error="offline"), {"git_sha": SHA_A})
    combined = watch._build_combined_watchdog_status(
        watchdog_status=clean, resolver_health={"issues": []}, code_health=code)
    assert combined["ok"] is False
    assert any("3 commit(s) behind" in e for e in combined["errors"])
    assert any("Could not fetch" in w for w in combined["warnings"])
    assert combined["code_identity"]["behind"] == 3


def test_watchdog_reads_the_reports_code_identity(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    assert watch._report_code_identity(path) is None
    path.write_text("{not json")
    assert watch._report_code_identity(path) is None
    path.write_text(json.dumps({"code_identity": "x"}))
    assert watch._report_code_identity(path) is None
    path.write_text(json.dumps({"code_identity": {"git_sha": SHA_A}}))
    assert watch._report_code_identity(path) == {"git_sha": SHA_A}


def test_evidence_report_records_the_code_that_ran(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(evidence_cycle, "source_identity", lambda: {"git_sha": SHA_A, "dirty": False})
    monkeypatch.setattr(evidence_cycle, "run_daily_cycle", lambda **_k: {})
    monkeypatch.setattr(evidence_cycle, "build_evidence_report", lambda: {"warning_flags": []})
    monkeypatch.setattr(evidence_cycle, "build_provider_telemetry_diagnostics", lambda limit=25: {"totals": {}})
    monkeypatch.setattr(evidence_cycle, "build_evidence_funnel_report", lambda **_k: {})
    payload = evidence_cycle.run_evidence_cycle(
        as_of=date(2026, 10, 20), report_dir=tmp_path / "d", weekly_report_dir=tmp_path / "w", write_weekly=False)
    written = json.loads(Path(payload["report_path"]).read_text())
    assert written["code_identity"] == {"git_sha": SHA_A, "dirty": False}


# ── backup manifest ─────────────────────────────────────────────────────────


def _state(tmp_path: Path) -> Path:
    state = tmp_path / ".options_calculator_pro"
    (state / "logs").mkdir(parents=True)
    (state / "logs" / "a.log").write_text("log\n")
    with sqlite3.connect(state / "ledger.sqlite") as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # as the live stores are
        conn.execute("PRAGMA user_version = 7")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO t VALUES (1)")
    conn.close()
    return state


def _backup(tmp_path: Path, state: Path) -> Path:
    out = tmp_path / "backups"
    assert backup_state.main(["--state-dir", str(state), "--output-dir", str(out)]) == 0
    [archive] = out.glob("*.tar.gz")
    return archive


def _manifest(archive: Path) -> dict:
    with tarfile.open(archive) as tf:
        return json.load(tf.extractfile(".options_calculator_pro/BACKUP_MANIFEST.json"))


def _rewrite(archive: Path, tmp_path: Path, edit) -> None:
    """Unpack, let ``edit`` change the tree, pack it back under the same name."""
    work = tmp_path / "rewrite"
    with tarfile.open(archive) as tf:
        tf.extractall(work, filter="data")
    edit(work / ".options_calculator_pro")
    archive.unlink()
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(work / ".options_calculator_pro", arcname=".options_calculator_pro")


def test_manifest_records_code_hashes_and_sqlite_facts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(backup_state, "source_identity", lambda: {"git_sha": SHA_A, "dirty": False})
    state = _state(tmp_path)
    manifest = _manifest(_backup(tmp_path, state))
    assert manifest["code_identity"] == {"git_sha": SHA_A, "dirty": False}
    assert manifest["manifest_version"] == 1 and manifest["archive_schema_version"] == 1
    files = {f["path"]: f for f in manifest["files"]}
    assert set(files) == {"logs/a.log", "ledger.sqlite"}
    import hashlib
    assert files["logs/a.log"]["sha256"] == hashlib.sha256(b"log\n").hexdigest()
    assert files["logs/a.log"]["bytes"] == 4 and "sqlite" not in files["logs/a.log"]
    facts = files["ledger.sqlite"]["sqlite"]
    assert facts["integrity_check"] == "ok" and facts["user_version"] == 7
    assert facts["schema_objects"] == 1 and len(facts["schema_sha256"]) == 64


def test_schema_hash_tracks_the_schema_not_the_rows(tmp_path: Path) -> None:
    def facts(path):
        return backup_state._sqlite_facts(path)
    db = tmp_path / "x.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE t (id INTEGER)")
    conn.close()
    first = facts(db)["schema_sha256"]
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO t VALUES (1)")
    conn.close()
    assert facts(db)["schema_sha256"] == first
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE INDEX i ON t (id)")
    conn.close()
    assert facts(db)["schema_sha256"] != first
    second = facts(db)["schema_sha256"]
    with sqlite3.connect(db) as conn:  # same object names, different definition
        conn.execute("ALTER TABLE t ADD COLUMN note TEXT")
    conn.close()
    assert facts(db)["schema_sha256"] != second


def test_a_snapshot_failing_integrity_check_is_archived_but_exits_3(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(backup_state, "_sqlite_facts",
                        lambda p: {"integrity_check": "row 3 missing from index", "user_version": 0,
                                   "schema_sha256": None})
    out = tmp_path / "backups"
    assert backup_state.main(["--state-dir", str(_state(tmp_path)), "--output-dir", str(out)]) == 3
    [archive] = out.glob("*.tar.gz")
    entry = next(f for f in _manifest(archive)["files"] if f["path"] == "ledger.sqlite")
    assert entry["sqlite"]["integrity_check"] == "row 3 missing from index"


def test_restore_verifies_and_keeps_the_manifest_as_provenance(tmp_path: Path, capsys) -> None:
    archive = _backup(tmp_path, _state(tmp_path))
    target = tmp_path / "restored"
    assert restore_state.main([str(archive), "--target-dir", str(target)]) == 0
    assert "manifest: 2 file(s) checked" in capsys.readouterr().out
    assert not (target / "BACKUP_MANIFEST.json").exists()
    kept = json.loads((target / "RESTORED_FROM_BACKUP_MANIFEST.json").read_text())
    assert {f["path"] for f in kept["files"]} == {"logs/a.log", "ledger.sqlite"}


def test_a_state_dir_manifest_from_a_restore_is_not_archived_as_this_backups(tmp_path: Path) -> None:
    state = _state(tmp_path)
    (state / "BACKUP_MANIFEST.json").write_text('{"stale": true}')
    manifest = _manifest(_backup(tmp_path, state))
    assert "stale" not in manifest
    assert "BACKUP_MANIFEST.json" not in {f["path"] for f in manifest["files"]}


@pytest.mark.parametrize("edit,needle", [
    (lambda root: (root / "logs" / "a.log").write_text("tampered\n"), "does not match manifest"),
    (lambda root: (root / "logs" / "extra.log").write_text("x"), "not in its manifest"),
    (lambda root: (root / "logs" / "a.log").unlink(), "missing from the archive"),
    (lambda root: (root / "BACKUP_MANIFEST.json").write_text("{nope"), "unreadable manifest"),
])
def test_restore_fails_closed_on_a_manifest_mismatch(tmp_path: Path, capsys, edit, needle) -> None:
    archive = _backup(tmp_path, _state(tmp_path))
    _rewrite(archive, tmp_path, edit)
    assert restore_state.main([str(archive), "--target-dir", str(tmp_path / "restored")]) == 2
    assert needle in capsys.readouterr().err


def test_archives_from_before_manifests_still_restore(tmp_path: Path, capsys) -> None:
    archive = _backup(tmp_path, _state(tmp_path))
    _rewrite(archive, tmp_path, lambda root: (root / "BACKUP_MANIFEST.json").unlink())
    assert restore_state.main([str(archive), "--target-dir", str(tmp_path / "restored")]) == 0
    assert "file hashes not verified" in capsys.readouterr().out


# ── /api/version and redundant indexes ─────────────────────────────────────


def test_version_endpoint_compares_the_served_bundle_with_the_running_api(monkeypatch, tmp_path: Path) -> None:
    import web.api.app as app_module
    monkeypatch.setattr(app_module, "project_root", tmp_path)
    monkeypatch.setattr(app_module, "_RUNNING_CODE", {"git_sha": SHA_A, "dirty": False})
    assert app_module.version()["frontend"] is None
    assert app_module.version()["frontend_matches_api"] is None

    dist = tmp_path / "web" / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "build-info.json").write_text(json.dumps({"git_sha": SHA_B, "dirty": False, "built_at": "t"}))
    out = app_module.version()
    assert out["api"]["git_sha"] == SHA_A and out["frontend"]["git_sha"] == SHA_B
    assert out["frontend_matches_api"] is False
    (dist / "build-info.json").write_text(json.dumps({"git_sha": SHA_A}))
    assert app_module.version()["frontend_matches_api"] is True
    monkeypatch.setattr(app_module, "_RUNNING_CODE", {"git_sha": None})
    assert app_module.version()["frontend_matches_api"] is None


def test_redundant_symbol_date_indexes_are_dropped(tmp_path: Path) -> None:
    from services.institutional_ml_db import InstitutionalMLDatabase
    db_path = tmp_path / "ml.db"
    InstitutionalMLDatabase(db_path=str(db_path))
    with sqlite3.connect(db_path) as conn:  # an older database still has them
        conn.execute("CREATE INDEX idx_earnings_events_symbol_date ON earnings_events (symbol, event_date)")
        conn.execute("CREATE INDEX idx_earnings_labels_symbol_date ON earnings_iv_decay_labels (symbol, event_date)")
    conn.close()
    InstitutionalMLDatabase(db_path=str(db_path))
    with sqlite3.connect(db_path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        plan = " ".join(str(r) for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM earnings_iv_decay_labels WHERE symbol='X' AND event_date='2026-01-01'"))
    conn.close()
    assert "idx_earnings_events_symbol_date" not in names
    assert "idx_earnings_labels_symbol_date" not in names
    assert "sqlite_autoindex_earnings_iv_decay_labels" in plan  # lookups still indexed
