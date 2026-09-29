"""Regressions for the second audit of #151.

F1. the seed script learned replay numbers under a paper trade's id;
F2. a claim_lost return or a failed claim renewal left a row 'finalizing' forever;
F3. a released row with exit facts sat in 'exited' forever;
F4. the backfill learned a crashed attempt's stale exit from a 'finalizing' row;
F5. invalidate() ignored learning written after a lost claim;
F6. an overflowing literal (1e999) read as inf;
F7. the NaN repair did nothing with no trades or in an unknown structure;
F8. a dry-run backfill created directories and lock files.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.backfill_prior_store_timestamps as backfill
import scripts.run_forward_loop as forward_loop
from scripts.seed_outcomes_from_replay import seed_from_trades
from services.baseline_evidence_store import BaselineEvidenceStore
from services.calibration_service import IVExpansionCalibration
from services.durable_json import PersistenceError, read_json_strict
from services.evidence_health import EvidenceHealthConfig, _check_evidence_integrity
from services.outcome_recorder import (
    LEARNING_WRITTEN_AFTER_CLAIM_LOST,
    OutcomeStore,
    finalize_trade_and_update_learning,
    make_trade_id,
)
from services.structure_prior_store import StructurePriorStore

T_MINUS_1 = date(2026, 4, 27)
EARNINGS = T_MINUS_1 + timedelta(days=1)


def _open(store: OutcomeStore, trade_id: str, *, source: str = "paper") -> None:
    store.insert_entry(
        trade_id=trade_id, symbol=trade_id, structure="otm_strangle",
        entry_date=EARNINGS - timedelta(days=6), earnings_date=EARNINGS,
        setup_score=0.6, source_type=source, entry_mid=2.0, execution_penalty_at_entry=0.0,
        notes=json.dumps({"pricing_context": {}}),
    )


def _expire(store: OutcomeStore, trade_id: str) -> None:
    with store._conn:
        store._conn.execute(
            "UPDATE outcome_trades SET finalizing_since = datetime('now', '-2 hours') WHERE trade_id = ?", (trade_id,)
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
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if "observation_ids" in raw:
        return sorted(raw["observation_ids"])
    return sorted(o["observation_id"] for e in raw["structures"].values() for o in e["observations"])


def _prior_returns(path: Path, trade_id: str) -> list[float]:
    raw = json.loads(path.read_text())
    return [o["realized_return_pct"] for e in raw["structures"].values() for o in e["observations"]
            if o["observation_id"] == trade_id]


def _run_day(tmp_path, store, day):
    return forward_loop.run_exit_detection(
        today=day, store=store, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: {"mid": 3.0, "context": {}},
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )


# ── F1 seed re-apply ──────────────────────────────────────────────────────────


def _replay_row(symbol):
    return {"pricing_source": "snapshot_replay", "trade_date": "2026-04-21", "event_date": "2026-04-28", "setup_score": 0.6, "gross_return_pct": 0.90,
            "net_return_pct": 0.80, "symbol": symbol, "structure": "otm_strangle"}


def _seed(tmp_path, rows):
    return seed_from_trades(rows, structure="otm_strangle", dry_run=False, outcome_store_path=tmp_path / "o.sqlite",
                            calibration_store_path=tmp_path / "cal.json", prior_store_path=tmp_path / "priors.json")


def test_seed_never_learns_replay_numbers_under_a_paper_trade(tmp_path, stores):
    entry, earn = date(2026, 4, 21), date(2026, 4, 28)
    store = OutcomeStore(tmp_path / "o.sqlite")
    paper = make_trade_id("AAA", entry, "otm_strangle", earnings_date=earn)
    store.insert_entry(trade_id=paper, symbol="AAA", structure="otm_strangle", entry_date=entry,
                       earnings_date=earn, setup_score=0.6, source_type="paper", entry_mid=2.0)
    invalid = make_trade_id("BBB", entry, "otm_strangle", earnings_date=earn)
    store.insert_entry(trade_id=invalid, symbol="BBB", structure="otm_strangle", entry_date=entry,
                       earnings_date=earn, setup_score=0.6, source_type="replay")
    store.update_exit(trade_id=invalid, exit_date=earn, realized_return_pct=1.0, realized_expansion_pct=1.0)
    store.mark_finalized(invalid)
    store.set_learning_update_status(invalid, "both_failed")
    store.invalidate(invalid, reason="bad replay data")

    summary = _seed(tmp_path, [_replay_row("AAA"), _replay_row("BBB")])

    assert summary["cal_updates"] == 0
    assert _ids(tmp_path / "cal.json") == [] and _ids(tmp_path / "priors.json") == []
    row = OutcomeStore(tmp_path / "o.sqlite").get_trade(paper)
    assert row["status"] == "open" and row["learning_update_status"] is None


def test_seed_relearns_its_own_failed_row_from_stored_values(tmp_path, stores):
    entry, earn = date(2026, 4, 21), date(2026, 4, 28)
    store = OutcomeStore(tmp_path / "o.sqlite")
    tid = make_trade_id("CCC", entry, "otm_strangle", earnings_date=earn)
    store.insert_entry(trade_id=tid, symbol="CCC", structure="otm_strangle", entry_date=entry,
                       earnings_date=earn, setup_score=0.6, source_type="replay")
    store.update_exit(trade_id=tid, exit_date=earn, realized_return_pct=12.0, realized_expansion_pct=7.0)
    store.mark_finalized(tid)
    store.set_learning_update_status(tid, "both_failed")

    _seed(tmp_path, [_replay_row("CCC")])  # this run's numbers are 80 / 90

    assert _ids(tmp_path / "priors.json") == [tid]
    assert _prior_returns(tmp_path / "priors.json", tid) == [12.0]
    assert OutcomeStore(tmp_path / "o.sqlite").get_trade(tid)["learning_update_status"] == "complete"


# ── F2 / F3 stuck rows are recovered ──────────────────────────────────────────


def test_claim_lost_after_learning_write_is_released_and_refinalized(tmp_path, stores):
    cal, _ = stores
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    real = cal.update

    def slow_write(*args, **kwargs):
        result = real(*args, **kwargs)
        _expire(store, "AAA")  # waited on the store lock past the lease
        return result

    cal.update = slow_write
    result = finalize_trade_and_update_learning(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=5.0,
                                                realized_expansion_pct=5.0, store=store)
    cal.update = real
    assert result["status"] == "claim_lost"
    row = store.get_trade("AAA")
    assert row["status"] == "exited" and row["finalizing_owner"] is None
    assert row["learning_update_status"] == LEARNING_WRITTEN_AFTER_CLAIM_LOST

    summary = _run_day(tmp_path, store, T_MINUS_1 + timedelta(days=1))

    assert summary["refinalized"] == 1
    row = store.get_trade("AAA")
    assert row["status"] == "finalized" and row["learning_update_status"] == "complete"
    assert row["exit_date"] == T_MINUS_1.isoformat() and row["realized_return_pct"] == 5.0
    assert _ids(tmp_path / "cal.json") == ["AAA"] and _ids(tmp_path / "priors.json") == ["AAA"]


def test_failed_claim_renewal_releases_the_row(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")

    def locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    store.renew_finalization_claim = locked
    with pytest.raises(sqlite3.OperationalError):
        finalize_trade_and_update_learning(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=5.0,
                                           realized_expansion_pct=5.0, store=store)
    del store.renew_finalization_claim
    row = store.get_trade("AAA")
    assert row["status"] == "exited" and row["finalizing_owner"] is None
    assert _ids(tmp_path / "cal.json") == []

    _run_day(tmp_path, store, T_MINUS_1 + timedelta(days=2))

    assert store.get_trade("AAA")["status"] == "finalized"
    assert _ids(tmp_path / "cal.json") == ["AAA"]


def test_crashed_finalizer_with_exit_is_refinalized_after_lease(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    assert store.claim_for_finalization("AAA", owner="crashed")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, exit_mid=2.5, realized_return_pct=25.0,
                      realized_expansion_pct=25.0, owner="crashed")

    # Live claim: left to its owner.
    assert _run_day(tmp_path, store, T_MINUS_1 + timedelta(days=1))["refinalized"] == 0
    assert store.get_trade("AAA")["status"] == "finalizing"

    _expire(store, "AAA")
    assert _run_day(tmp_path, store, T_MINUS_1 + timedelta(days=1))["refinalized"] == 1
    row = store.get_trade("AAA")
    assert row["status"] == "finalized" and row["realized_return_pct"] == 25.0


def test_sweep_leaves_rows_without_realized_values_and_invalid_rows(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "NOVAL")
    store.update_exit(trade_id="NOVAL", exit_date=T_MINUS_1, exit_mid=2.5)  # record_trade_exit step 1 only
    _open(store, "BAD")
    store.update_exit(trade_id="BAD", exit_date=T_MINUS_1, realized_return_pct=1.0, realized_expansion_pct=1.0)
    store.invalidate("BAD", reason="x")
    _open(store, "TODAY")

    day = T_MINUS_1 + timedelta(days=1)
    assert store.trades_pending_finalization(day) == []
    assert _run_day(tmp_path, store, day)["refinalized"] == 0
    assert _ids(tmp_path / "cal.json") == []


def test_dry_run_does_not_refinalize(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=1.0, realized_expansion_pct=1.0)
    forward_loop.run_exit_detection(
        today=EARNINGS, store=store, log_path=tmp_path / "log.jsonl", dry_run=True,
        price_fetcher=lambda **_: {"mid": 3.0, "context": {}},
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )
    assert store.get_trade("AAA")["status"] == "exited"


def test_health_flags_exits_that_were_never_finalized(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=1.0, realized_expansion_pct=1.0)
    cfg = EvidenceHealthConfig(expected_date=EARNINGS, outcome_store_path=tmp_path / "o.sqlite",
                               baseline_store_path=tmp_path / "b.sqlite")
    now = datetime.combine(EARNINGS + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    result = _check_evidence_integrity(cfg, now)
    assert result["summary"]["selector"]["exited_not_finalized"] == ["AAA"]
    assert any("never finalized" in issue["message"] for issue in result["issues"])


# ── F4 backfill and in-flight rows ────────────────────────────────────────────


def _backfill(tmp_path, *extra):
    return backfill.main(["--db-path", str(tmp_path / "o.sqlite"), "--prior-store", str(tmp_path / "priors.json"),
                          "--cal-store", str(tmp_path / "cal.json"), *extra])


def test_backfill_and_retry_agree_on_a_crashed_attempts_exit(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    assert store.claim_for_finalization("AAA", owner="crashed")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, exit_mid=3.8, realized_return_pct=90.0,
                      realized_expansion_pct=90.0, owner="crashed")
    _expire(store, "AAA")

    # A backfill while the row is in flight and unlearned adds nothing ...
    assert _backfill(tmp_path, "--target", "production") == 0
    assert _ids(tmp_path / "cal.json") == [] and _ids(tmp_path / "priors.json") == []

    # ... and a retry carrying other numbers learns the RECORDED exit, so the
    # row, calibration and priors all agree.
    result = finalize_trade_and_update_learning(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=-30.0,
                                                realized_expansion_pct=-30.0, store=store)
    assert result["learning_update_status"] == "complete"
    row = store.get_trade("AAA")
    assert row["realized_return_pct"] == 90.0 and row["exit_mid"] == 3.8
    assert _prior_returns(tmp_path / "priors.json", "AAA") == [90.0]
    assert json.loads((tmp_path / "cal.json").read_text())["expansions"] == [90.0]


def test_backfill_keeps_an_in_flight_rows_learned_observation(tmp_path, stores):
    cal, priors = stores
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    assert store.claim_for_finalization("AAA", owner="worker")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=4.0, realized_expansion_pct=4.0,
                      owner="worker")
    cal.update(0.6, 4.0, observation_id="AAA", source_type="paper", observation_date=T_MINUS_1)
    priors.update(structure="otm_strangle", realized_return_pct=4.0, realized_expansion_pct=4.0,
                  source_type="paper", observation_date=T_MINUS_1, observation_id="AAA")

    assert _backfill(tmp_path, "--target", "production") == 0

    assert _ids(tmp_path / "priors.json") == ["AAA"] and _ids(tmp_path / "cal.json") == ["AAA"]


# ── F5 invalidate after a lost claim ──────────────────────────────────────────


def test_invalidate_reports_learning_written_after_lost_claim(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.flag_learning_written_without_claim("AAA")
    assert store.invalidate("AAA", reason="x")["learning_already_applied"] is True


# ── F6 overflowing literals ───────────────────────────────────────────────────


@pytest.mark.parametrize("literal", ["1e999", "-1e999", "NaN", "Infinity"])
def test_strict_read_rejects_overflowing_and_non_finite_literals(tmp_path, literal):
    path = tmp_path / "s.json"
    path.write_text('{"values": [1.5, ' + literal + ']}')
    with pytest.raises(PersistenceError, match="non-finite"):
        read_json_strict(path)


def test_strict_read_keeps_ordinary_floats_and_strings(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"values": [1.5, 1e300, -2.0], "label": "NaN"}')
    assert read_json_strict(path) == {"values": [1.5, 1e300, -2.0], "label": "NaN"}


# ── F7 NaN repair edge cases ──────────────────────────────────────────────────


def test_repair_cleans_calibration_when_there_are_no_trades(tmp_path):
    OutcomeStore(tmp_path / "o.sqlite")
    (tmp_path / "cal.json").write_text(
        '{"schema_version": 2, "scores": [0.5, 0.7], "expansions": [NaN, 3.0], "sources": ["paper", "replay"], '
        '"timestamps": ["2026-01-01", "2026-01-02"], "observation_ids": ["OLD", "OK"], "n": 2}'
    )
    assert _backfill(tmp_path, "--target", "production") == 0
    cal = read_json_strict(tmp_path / "cal.json")
    assert cal["scores"] == [0.7] and cal["expansions"] == [3.0]
    assert cal["sources"] == ["replay"] and cal["timestamps"] == ["2026-01-02"] and cal["n"] == 1


def test_repair_keeps_calibration_when_there_are_no_trades(tmp_path):
    OutcomeStore(tmp_path / "o.sqlite")
    original = ('{"schema_version": 2, "scores": [0.5], "expansions": [3.0], "sources": ["paper"], '
                '"timestamps": ["2026-01-01"], "observation_ids": ["OLD"], "n": 1}')
    (tmp_path / "cal.json").write_text(original)
    assert _backfill(tmp_path, "--target", "production") == 0
    assert (tmp_path / "cal.json").read_text() == original


def test_repair_cleans_unknown_structures(tmp_path, stores):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open(store, "AAA")
    store.update_exit(trade_id="AAA", exit_date=T_MINUS_1, realized_return_pct=5.0, realized_expansion_pct=5.0)
    store.mark_finalized("AAA")
    (tmp_path / "priors.json").write_text(json.dumps({"schema_version": 2, "structures": {"legacy_butterfly": {
        "structure": "legacy_butterfly", "observations": [
            {"observation_id": "X", "observation_date": "2026-01-01", "source_type": "paper",
             "realized_return_pct": float("nan"), "realized_expansion_pct": 1.0},
            {"observation_id": "Y", "observation_date": "2026-01-02", "source_type": "paper",
             "realized_return_pct": 2.0, "realized_expansion_pct": 1.0},
        ]}}}))

    assert _backfill(tmp_path, "--target", "production") == 0

    prior = read_json_strict(tmp_path / "priors.json")
    legacy = prior["structures"]["legacy_butterfly"]
    assert [o["observation_id"] for o in legacy["observations"]] == ["Y"]
    assert legacy["observation_count"] == 1
    assert _prior_returns(tmp_path / "priors.json", "AAA") == [5.0]


# ── F8 dry run touches nothing ────────────────────────────────────────────────


def test_dry_run_backfill_creates_no_files(tmp_path):
    OutcomeStore(tmp_path / "o.sqlite")
    prior = tmp_path / "new_p" / "priors.json"
    cal = tmp_path / "new_c" / "cal.json"
    assert backfill.main(["--db-path", str(tmp_path / "o.sqlite"), "--prior-store", str(prior),
                          "--cal-store", str(cal)]) == 0
    assert not (tmp_path / "new_p").exists() and not (tmp_path / "new_c").exists()


def test_seed_leaves_a_finalized_paper_trade_with_failed_learning_to_its_own_retry(tmp_path, stores):
    entry, earn = date(2026, 4, 21), date(2026, 4, 28)
    store = OutcomeStore(tmp_path / "o.sqlite")
    tid = make_trade_id("DDD", entry, "otm_strangle", earnings_date=earn)
    store.insert_entry(trade_id=tid, symbol="DDD", structure="otm_strangle", entry_date=entry,
                       earnings_date=earn, setup_score=0.6, source_type="paper", entry_mid=2.0)
    store.update_exit(trade_id=tid, exit_date=earn, realized_return_pct=-10.0, realized_expansion_pct=-5.0)
    store.mark_finalized(tid)
    store.set_learning_update_status(tid, "both_failed")

    _seed(tmp_path, [_replay_row("DDD")])

    assert _ids(tmp_path / "cal.json") == [] and _ids(tmp_path / "priors.json") == []
    assert OutcomeStore(tmp_path / "o.sqlite").get_trade(tid)["learning_update_status"] == "both_failed"
