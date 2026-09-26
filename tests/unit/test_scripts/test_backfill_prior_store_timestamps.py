"""Tests for scripts/backfill_prior_store_timestamps.py (previously untested)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import scripts.backfill_prior_store_timestamps as backfill
from services.outcome_recorder import OutcomeStore

EXIT = date(2026, 4, 27)


def _finalized(store, trade_id, structure="otm_strangle", ret=10.0, *, exit_date=EXIT):
    store.insert_entry(
        trade_id=trade_id, symbol=trade_id, structure=structure,
        entry_date=exit_date - timedelta(days=5), earnings_date=exit_date + timedelta(days=1),
        setup_score=0.6, source_type="paper", entry_mid=2.0,
    )
    store.update_exit(trade_id=trade_id, exit_date=exit_date, exit_mid=2.0,
                      realized_return_pct=ret, realized_expansion_pct=ret / 2)
    store.mark_finalized(trade_id)


def _store(tmp_path):
    store = OutcomeStore(tmp_path / "outcomes.sqlite")
    _finalized(store, "MU", ret=9.0)
    _finalized(store, "NVDA", ret=-10.0)
    _finalized(store, "SPY-condor", structure="iron_condor", ret=4.0)
    # Invalidated after finalization (the case this script now cleans up).
    _finalized(store, "AMZN", ret=900.0)
    store.invalidate("AMZN", reason="exit repriced an unheld strike")
    # Legacy notes flag, set by hand after finalization like the real AMZN/TTD.
    _finalized(store, "TTD", ret=-500.0)
    store._conn.execute(
        "UPDATE outcome_trades SET notes = ? WHERE trade_id = 'TTD'",
        (json.dumps({"evidence_invalidated": {"reason_code": "x", "reason": "y"}}),),
    )
    store._conn.commit()
    # Exited with a return but never finalized (learning never applied).
    store.insert_entry(trade_id="EXITED", symbol="EXITED", structure="otm_strangle",
                       entry_date=EXIT - timedelta(days=5), earnings_date=EXIT + timedelta(days=1),
                       setup_score=0.6, source_type="paper", entry_mid=2.0)
    store.update_exit(trade_id="EXITED", exit_date=EXIT, exit_mid=3.0,
                      realized_return_pct=50.0, realized_expansion_pct=25.0)
    # Not finalized.
    store.insert_entry(trade_id="OPEN", symbol="OPEN", structure="otm_strangle",
                       entry_date=EXIT, earnings_date=EXIT + timedelta(days=9),
                       setup_score=0.6, source_type="paper", entry_mid=2.0)
    store.close()
    return tmp_path / "outcomes.sqlite"


def test_fetch_returns_only_valid_finalized_trades(tmp_path):
    db = _store(tmp_path)

    trades = backfill._fetch_trades(db)

    assert sorted(t["trade_id"] for t in trades) == ["MU", "NVDA", "SPY-condor"]
    assert backfill._fetch_invalidated_trade_ids(db) == {"AMZN", "TTD"}


def test_production_rebuild_keeps_condors_and_purges_invalidated_observations(tmp_path):
    db = _store(tmp_path)
    prior_path = tmp_path / "priors.json"
    cal_path = tmp_path / "calibration.json"
    # A store contaminated before invalidation, plus a structure the script
    # has no trades for.
    prior_path.write_text(json.dumps({
        "schema_version": 2,
        "structures": {
            "otm_strangle": {"structure": "otm_strangle", "schema_version": 2, "observations": [
                {"observation_id": "AMZN", "observation_date": "2026-04-27", "source_type": "paper",
                 "realized_return_pct": 900.0, "realized_expansion_pct": 450.0},
            ], "observation_count": 1},
            "call_calendar": {"structure": "call_calendar", "schema_version": 2, "observations": [
                {"observation_id": "TTD", "observation_date": "2026-04-27", "source_type": "paper",
                 "realized_return_pct": -500.0, "realized_expansion_pct": -250.0},
                {"observation_id": "CAL-1", "observation_date": "2026-04-20", "source_type": "paper",
                 "realized_return_pct": 3.0, "realized_expansion_pct": 1.5},
            ], "observation_count": 2},
            "future_structure": {"structure": "future_structure", "observations": []},
        },
    }))

    assert backfill.main([
        "--db-path", str(db), "--prior-store", str(prior_path), "--cal-store", str(cal_path),
        "--target", "production",
    ]) == 0

    priors = json.loads(prior_path.read_text())["structures"]
    strangle_ids = {o["observation_id"] for o in priors["otm_strangle"]["observations"]}
    assert strangle_ids == {"MU", "NVDA"}
    assert priors["otm_strangle"]["avg_return_pct"] == -0.5
    assert [o["observation_id"] for o in priors["iron_condor"]["observations"]] == ["SPY-condor"]
    assert [o["observation_id"] for o in priors["call_calendar"]["observations"]] == ["CAL-1"]
    assert priors["call_calendar"]["observation_count"] == 1
    assert "future_structure" in priors

    calibration = json.loads(cal_path.read_text())
    assert calibration["n"] == 3
    assert sorted(calibration["observation_ids"]) == ["MU", "NVDA", "SPY-condor"]
    assert prior_path.with_suffix(".json.pre_migration_backup").exists()


def test_dry_run_writes_nothing(tmp_path):
    db = _store(tmp_path)
    prior_path = tmp_path / "priors.json"
    cal_path = tmp_path / "calibration.json"

    assert backfill.main(["--db-path", str(db), "--prior-store", str(prior_path), "--cal-store", str(cal_path)]) == 0

    assert not prior_path.exists()
    assert not cal_path.exists()


def test_missing_database_fails_cleanly(tmp_path):
    assert backfill.main(["--db-path", str(tmp_path / "nope.sqlite")]) == 1
