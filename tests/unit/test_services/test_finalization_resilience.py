"""Regressions for an independent audit of #150 (main at 0bfefbd).

1. one claimed/failing trade stopped the whole exit run;
2. a failed exit write left a row 'finalizing' forever;
3. the prior/calibration backfill bypassed the store locks (lost updates);
4. re-finalizing wrote the caller's numbers, not the finalized facts;
5. a worker past its lease still wrote learning;
6. the seed script could drop a trade from learning permanently;
7. a legacy NaN in a store file blocked every later update;
8. a directory-fsync failure after the replace was reported as "nothing written".
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.backfill_prior_store_timestamps as backfill
import scripts.run_forward_loop as forward_loop
from scripts.seed_outcomes_from_replay import seed_from_trades
from services.baseline_evidence_store import BaselineEvidenceStore
from services.calibration_service import IVExpansionCalibration
from services.durable_json import PersistenceError
from services.evidence_health import EvidenceHealthConfig, _check_evidence_integrity
from services.outcome_recorder import (
    LEARNING_WRITTEN_AFTER_CLAIM_LOST,
    OutcomeStore,
    finalize_trade_and_update_learning,
)
from services.structure_prior_store import StructurePriorStore

T_MINUS_1 = date(2026, 4, 27)
EARNINGS = T_MINUS_1 + timedelta(days=1)


def _open(store: OutcomeStore, trade_id: str, *, earnings: date = EARNINGS) -> None:
    store.insert_entry(
        trade_id=trade_id, symbol=trade_id, structure="otm_strangle",
        entry_date=earnings - timedelta(days=6), earnings_date=earnings,
        setup_score=0.6, source_type="paper", entry_mid=2.0, execution_penalty_at_entry=0.0,
        notes=json.dumps({"pricing_context": {}}),
    )


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    import services.calibration_service as calibration_service
    import services.structure_prior_store as structure_prior_store

    cal = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    priors = StructurePriorStore(tmp_path / "priors.json")
    monkeypatch.setattr(calibration_service, "get_calibration", lambda: cal)
    monkeypatch.setattr(structure_prior_store, "get_structure_prior_store", lambda: priors)
    return cal, priors


def _ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    if "observation_ids" in raw:
        return sorted(raw["observation_ids"])
    return sorted(o["observation_id"] for e in raw["structures"].values() for o in e["observations"])


# ── 1. one trade cannot stop the exit run ─────────────────────────────────────


def _exit(tmp_path, store, **kwargs):
    return forward_loop.run_exit_detection(
        today=T_MINUS_1, store=store, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: {"mid": 3.0, "context": {}},
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"), **kwargs,
    )


def test_a_live_claim_is_skipped_and_the_run_continues(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    _open(store, "BBB")
    assert store.claim_for_finalization("AAA", owner="other-worker")

    summary = _exit(tmp_path, store)

    # Not even attempted: a live claim is excluded from the due list.
    assert (summary["exits"], summary["skipped"]) == (1, 0)
    assert store.get_trade("AAA")["status"] == "finalizing"
    assert store.get_trade("BBB")["status"] == "finalized"


def test_a_finalizer_error_on_one_trade_does_not_stop_the_run(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    _open(store, "BBB")
    done = []

    def finalizer(**kwargs):
        if kwargs["trade_id"] == "AAA":
            raise sqlite3.OperationalError("database is locked")
        done.append(kwargs["trade_id"])
        return {"status": "finalized"}

    summary = _exit(tmp_path, store, finalizer=finalizer)

    assert done == ["BBB"]
    assert (summary["exits"], summary["skipped"]) == (1, 1)
    assert store.get_trade("AAA")["last_exit_attempt_reason"] == "finalize_failed: OperationalError"


# ── 2. a failed exit write releases the claim ─────────────────────────────────


def test_failed_exit_write_releases_the_claim_and_the_row_is_swept(tmp_path, stores, monkeypatch):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")

    def locked(**_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "update_exit", locked)
    with pytest.raises(sqlite3.OperationalError):
        finalize_trade_and_update_learning(
            trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=5.0, realized_expansion_pct=5.0, store=store,
        )

    row = store.get_trade("AAA")
    assert (row["status"], row["finalizing_owner"]) == ("open", None)
    assert [r["trade_id"] for r in store.mark_missing_exits(EARNINGS)] == ["AAA"]
    assert store.get_trade("AAA")["status"] == "exit_missing"


def test_a_crashed_claim_without_exit_is_swept_after_its_lease(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    assert store.claim_for_finalization("AAA", owner="crashed")
    assert store.mark_missing_exits(EARNINGS) == []  # live claim: left alone
    store._conn.execute("UPDATE outcome_trades SET finalizing_since = datetime('now', '-2 hours')")
    store._conn.commit()

    assert [r["trade_id"] for r in store.mark_missing_exits(EARNINGS)] == ["AAA"]
    row = store.get_trade("AAA")
    assert (row["status"], row["finalizing_owner"]) == ("exit_missing", None)


# ── 3. backfill vs a concurrent finalize in another process ───────────────────


def _finalize_bbb(db: str, started, finished) -> None:
    from services.outcome_recorder import OutcomeStore as Store
    from services.outcome_recorder import finalize_trade_and_update_learning as finalize

    started.set()
    finalize(trade_id="BBB", exit_date=T_MINUS_1, realized_return_pct=7.0, realized_expansion_pct=7.0,
             store=Store(Path(db)))
    finished.set()


def test_backfill_cannot_erase_a_concurrent_finalize(tmp_path, monkeypatch):
    db, cal, prior = tmp_path / "o.sqlite", tmp_path / "cal.json", tmp_path / "priors.json"
    monkeypatch.setenv("OPTIONS_CALCULATOR_CALIBRATION_PATH", str(cal))
    monkeypatch.setenv("OPTIONS_CALCULATOR_PRIORS_PATH", str(prior))
    store = OutcomeStore(db)
    _open(store, "AAA")
    _open(store, "BBB")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, exit_mid=2.5, realized_return_pct=5.0,
                      realized_expansion_pct=5.0)
    store.mark_finalized("AAA")
    IVExpansionCalibration(store_path=cal).update(0.6, 5.0, observation_id="AAA", observation_date=T_MINUS_1)

    ctx = mp.get_context("spawn")
    started, finished = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_finalize_bbb, args=(str(db), started, finished))
    real_backup = backfill._backup

    def backup_while_other_process_finalizes(path, dry_run):
        # Inside the backfill's locks, after it fetched trades: another
        # process tries to finalize BBB and must wait for the lock.
        if not child.is_alive() and not started.is_set():
            child.start()
            assert started.wait(60)
            time.sleep(1.0)
            assert not finished.is_set(), "finalize wrote learning while the backfill held the locks"
        real_backup(path, dry_run)

    monkeypatch.setattr(backfill, "_backup", backup_while_other_process_finalizes)
    assert backfill.main(["--db-path", str(db), "--prior-store", str(prior), "--cal-store", str(cal),
                          "--target", "production"]) == 0
    child.join(timeout=120)
    assert child.exitcode == 0

    assert _ids(cal) == ["AAA", "BBB"]
    assert _ids(prior) == ["AAA", "BBB"]
    assert OutcomeStore(db).get_trade("BBB")["status"] == "finalized"


def test_backfill_includes_in_flight_finalizations(tmp_path):
    db = tmp_path / "o.sqlite"
    store = OutcomeStore(db)
    _open(store, "FLY")
    assert store.claim_for_finalization("FLY", owner="w")
    store.update_exit(trade_id="FLY", exit_date=T_MINUS_1, exit_mid=2.5, realized_return_pct=5.0,
                      realized_expansion_pct=5.0, owner="w")

    assert [t["trade_id"] for t in backfill._fetch_trades(db)] == ["FLY"]


# ── 4. re-finalizing uses the finalized facts ─────────────────────────────────


def test_refinalize_learns_the_stored_values_not_the_callers(tmp_path, stores, monkeypatch):
    cal, priors = stores
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    real_update = cal.update
    monkeypatch.setattr(cal, "update", lambda *a, **k: (_ for _ in ()).throw(PersistenceError("disk full")))
    first = finalize_trade_and_update_learning(
        trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=50.0, realized_expansion_pct=50.0, store=store,
    )
    assert first["learning_update_status"] == "calibration_failed"
    monkeypatch.setattr(cal, "update", real_update)

    retry = finalize_trade_and_update_learning(
        trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=-75.0, realized_expansion_pct=-75.0, store=store,
    )

    assert retry["learning_update_status"] == "complete"
    assert any("stored realized values" in w for w in retry["warnings"])
    assert cal._expansions == [50.0]
    assert json.loads((tmp_path / "priors.json").read_text())["structures"]["otm_strangle"]["observations"][0][
        "realized_return_pct"] == 50.0


# ── 5. a lapsed claim does not write, or is flagged if it did ─────────────────


def test_a_claim_taken_over_before_learning_writes_nothing(tmp_path, stores, monkeypatch):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    real_update_exit = store.update_exit

    def exit_then_lose_the_claim(**kwargs):
        written = real_update_exit(**kwargs)
        store._conn.execute("UPDATE outcome_trades SET finalizing_since = datetime('now', '-2 hours')")
        store._conn.commit()
        store.invalidate("AAA", reason="bad quote")  # allowed once the lease has expired
        return written

    monkeypatch.setattr(store, "update_exit", exit_then_lose_the_claim)
    result = finalize_trade_and_update_learning(
        trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=5.0, realized_expansion_pct=5.0, store=store,
    )

    assert result["status"] == "claim_lost"
    assert not (tmp_path / "cal.json").exists()
    assert not (tmp_path / "priors.json").exists()


def test_learning_written_after_the_claim_lapsed_is_flagged_for_repair(tmp_path, stores, monkeypatch):
    cal, _ = stores
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    real_update = cal.update

    def slow_calibration_write(*args, **kwargs):
        # The write waited on the store lock past the lease; meanwhile the row
        # was invalidated.
        store._conn.execute("UPDATE outcome_trades SET finalizing_since = datetime('now', '-2 hours')")
        store._conn.commit()
        store.invalidate("AAA", reason="bad quote")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(cal, "update", slow_calibration_write)
    result = finalize_trade_and_update_learning(
        trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=5.0, realized_expansion_pct=5.0, store=store,
    )

    assert result["status"] == "claim_lost"
    assert store.get_trade("AAA")["learning_update_status"] == LEARNING_WRITTEN_AFTER_CLAIM_LOST
    assert not (tmp_path / "priors.json").exists()  # the prior write was not attempted
    health = _check_evidence_integrity(
        EvidenceHealthConfig(expected_date=EARNINGS, outcome_store_path=tmp_path / "o.sqlite",
                             baseline_store_path=tmp_path / "b.sqlite",
                             calibration_store_path=tmp_path / "cal.json",
                             prior_store_path=tmp_path / "priors.json"),
        datetime(2026, 4, 29, tzinfo=timezone.utc),
    )
    assert health["summary"]["selector"]["invalidated_after_learning"] == ["AAA"]


# ── 6. seed script failures are retryable ─────────────────────────────────────


def _seed(tmp_path):
    rows = [{"pricing_source": "snapshot_replay", "trade_date": "2025-01-10", "event_date": "2025-01-15", "setup_score": 0.6,
             "gross_return_pct": 0.05, "net_return_pct": 0.04, "symbol": "AAA"}]
    return seed_from_trades(
        rows, structure="atm_straddle", dry_run=False,
        outcome_store_path=tmp_path / "o.sqlite", calibration_store_path=tmp_path / "cal.json",
        prior_store_path=tmp_path / "priors.json",
    )


def test_seed_learning_failure_is_recorded_and_completed_on_rerun(tmp_path, monkeypatch):
    def disk_full(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("services.durable_json.os.replace", disk_full)
    _seed(tmp_path)
    row = next(iter(OutcomeStore(tmp_path / "o.sqlite").list_for_diagnostics()))
    assert (row["status"], row["learning_update_status"]) == ("finalized", "both_failed")
    monkeypatch.undo()

    _seed(tmp_path)

    row = next(iter(OutcomeStore(tmp_path / "o.sqlite").list_for_diagnostics()))
    assert row["learning_update_status"] == "complete"
    assert _ids(tmp_path / "cal.json") == [row["trade_id"]]
    assert _ids(tmp_path / "priors.json") == [row["trade_id"]]


# ── 7. legacy NaN files fail clearly and are repairable ───────────────────────


def test_legacy_nan_file_is_refused_with_a_repair_hint_and_backfill_repairs_it(tmp_path):
    cal_path, prior_path, db = tmp_path / "cal.json", tmp_path / "priors.json", tmp_path / "o.sqlite"
    cal_path.write_text('{"schema_version": 2, "scores": [0.5], "expansions": [NaN], "sources": ["paper"], '
                        '"timestamps": ["2026-01-01"], "observation_ids": ["OLD"], "n": 1}')
    prior_path.write_text(json.dumps({"schema_version": 2, "structures": {"put_calendar": {
        "structure": "put_calendar", "schema_version": 2, "observation_count": 1,
        "observations": [{"observation_id": "OLD", "observation_date": "2026-01-01", "source_type": "paper",
                          "realized_return_pct": float("nan"), "realized_expansion_pct": 1.0}]}}}))

    with pytest.raises(PersistenceError, match="backfill_prior_store_timestamps"):
        IVExpansionCalibration(store_path=cal_path).update(0.6, 1.0, observation_id="NEW", observation_date=T_MINUS_1)

    store = OutcomeStore(db)
    _open(store, "AAA")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, exit_mid=2.5, realized_return_pct=5.0,
                      realized_expansion_pct=5.0)
    store.mark_finalized("AAA")
    assert backfill.main(["--db-path", str(db), "--prior-store", str(prior_path), "--cal-store", str(cal_path),
                          "--target", "production"]) == 0

    assert IVExpansionCalibration(store_path=cal_path).update(0.6, 1.0, observation_id="NEW",
                                                              observation_date=T_MINUS_1) is True
    assert StructurePriorStore(prior_path).update(
        structure="put_calendar", realized_return_pct=1.0, realized_expansion_pct=1.0,
        observation_date=T_MINUS_1, observation_id="NEW2") is True


# ── 8. directory fsync after the replace ──────────────────────────────────────


def test_directory_fsync_failure_after_replace_is_not_reported_as_unwritten(tmp_path, monkeypatch):
    import services.durable_json as durable_json

    store = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    real_open = durable_json.os.open

    def failing_dir_open(path, flags, *args):
        if Path(path) == tmp_path:
            raise OSError(22, "Invalid argument")
        return real_open(path, flags, *args)

    monkeypatch.setattr(durable_json.os, "open", failing_dir_open)
    assert store.update(0.5, 5.0, observation_id="A", observation_date=T_MINUS_1) is True

    assert store._n() == 1
    assert IVExpansionCalibration(store_path=tmp_path / "cal.json")._n() == 1
