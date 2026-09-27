"""Learning evidence must be durable, never silently lost, and finalized once.

Regressions for an external audit of main at b17d510:
1. two writers (processes or stale store instances) lost each other's
   calibration/prior observations;
2. a failed write was reported as a successful learning update;
3. two workers could both "claim" and finalize the same trade.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from datetime import date
from pathlib import Path

import pytest

from services.calibration_service import IVExpansionCalibration
from services.durable_json import PersistenceError
from services.outcome_recorder import OutcomeStore, finalize_trade_and_update_learning
from services.structure_prior_store import StructurePriorStore

DAY = date(2026, 5, 1)


def _cal_ids(path: Path) -> list[str]:
    return sorted(json.loads(path.read_text())["observation_ids"])


def _prior_ids(path: Path, structure: str = "otm_strangle") -> list[str]:
    raw = json.loads(path.read_text())
    return sorted(o["observation_id"] for o in raw["structures"][structure]["observations"])


def _prior_update(store: StructurePriorStore, obs_id: str, ret: float = 1.0):
    return store.update(
        structure="otm_strangle", realized_return_pct=ret, realized_expansion_pct=ret,
        source_type="paper", observation_date=DAY, observation_id=obs_id,
    )


# ── 1. no lost updates ────────────────────────────────────────────────────────


def test_stale_calibration_instances_do_not_lose_each_others_observations(tmp_path):
    path = tmp_path / "cal.json"
    a, b = IVExpansionCalibration(store_path=path), IVExpansionCalibration(store_path=path)

    assert a.update(0.5, 5.0, observation_id="A", observation_date=DAY) is True
    assert b.update(0.6, 6.0, observation_id="B", observation_date=DAY) is True

    assert _cal_ids(path) == ["A", "B"]
    assert IVExpansionCalibration(store_path=path)._n() == 2
    assert b._n() == 2  # the writer adopts the merged state


def test_stale_prior_instances_do_not_lose_each_others_observations(tmp_path):
    path = tmp_path / "priors.json"
    a, b = StructurePriorStore(path), StructurePriorStore(path)

    assert _prior_update(a, "A") is True
    assert _prior_update(b, "B") is True

    assert _prior_ids(path) == ["A", "B"]
    assert json.loads(path.read_text())["structures"]["otm_strangle"]["observation_count"] == 2


def test_duplicate_is_detected_against_disk_not_stale_memory(tmp_path):
    cal_path, prior_path = tmp_path / "cal.json", tmp_path / "priors.json"
    cal_a, cal_b = IVExpansionCalibration(store_path=cal_path), IVExpansionCalibration(store_path=cal_path)
    prior_a, prior_b = StructurePriorStore(prior_path), StructurePriorStore(prior_path)

    assert cal_a.update(0.5, 5.0, observation_id="T1", observation_date=DAY) is True
    assert cal_b.update(0.5, 5.0, observation_id="T1", observation_date=DAY) is False
    assert _prior_update(prior_a, "T1") is True
    assert _prior_update(prior_b, "T1") is False

    assert _cal_ids(cal_path) == ["T1"]
    assert _prior_ids(prior_path) == ["T1"]


def test_save_on_a_stale_instance_never_overwrites_newer_evidence(tmp_path):
    cal_path, prior_path = tmp_path / "cal.json", tmp_path / "priors.json"
    stale_cal = IVExpansionCalibration(store_path=cal_path)
    stale_prior = StructurePriorStore(prior_path)
    IVExpansionCalibration(store_path=cal_path).update(0.5, 5.0, observation_id="A", observation_date=DAY)
    _prior_update(StructurePriorStore(prior_path), "A")

    stale_cal.save()
    stale_prior.save()

    assert _cal_ids(cal_path) == ["A"]
    assert _prior_ids(prior_path) == ["A"]


def _cal_writer(path: str, worker: int, count: int) -> None:
    store = IVExpansionCalibration(store_path=Path(path))
    for i in range(count):
        store.update(0.5, float(i), observation_id=f"w{worker}-{i}", observation_date=DAY)


def _prior_writer(path: str, worker: int, count: int) -> None:
    store = StructurePriorStore(Path(path))
    for i in range(count):
        _prior_update(store, f"w{worker}-{i}", float(i))


@pytest.mark.parametrize(("writer", "ids"), [(_cal_writer, _cal_ids), (_prior_writer, _prior_ids)],
                         ids=["calibration", "priors"])
def test_concurrent_processes_lose_no_observations(tmp_path, writer, ids):
    path = tmp_path / "store.json"
    workers, per_worker = 4, 15
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=writer, args=(str(path), w, per_worker)) for w in range(workers)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
        assert proc.exitcode == 0

    expected = sorted(f"w{w}-{i}" for w in range(workers) for i in range(per_worker))
    assert ids(path) == expected


# ── 2. failures are not acknowledged ──────────────────────────────────────────


def _unwritable(tmp_path: Path, name: str) -> Path:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    return blocker / name


def test_unwritable_calibration_update_raises_and_keeps_memory_clean(tmp_path):
    store = IVExpansionCalibration(store_path=_unwritable(tmp_path, "cal.json"))

    with pytest.raises(PersistenceError):
        store.update(0.5, 5.0, observation_id="Z", observation_date=DAY)

    assert store._n() == 0


def test_unwritable_prior_update_raises_and_keeps_memory_clean(tmp_path):
    store = StructurePriorStore(_unwritable(tmp_path, "priors.json"))

    with pytest.raises(PersistenceError):
        _prior_update(store, "Z")

    assert store._data == {}


def test_corrupt_store_file_is_refused_not_overwritten(tmp_path):
    cal_path, prior_path = tmp_path / "cal.json", tmp_path / "priors.json"
    cal_path.write_text("{truncated")
    prior_path.write_text("{truncated")

    with pytest.raises(PersistenceError):
        IVExpansionCalibration(store_path=cal_path).update(0.5, 5.0, observation_id="Z", observation_date=DAY)
    with pytest.raises(PersistenceError):
        _prior_update(StructurePriorStore(prior_path), "Z")

    assert cal_path.read_text() == "{truncated"
    assert prior_path.read_text() == "{truncated"


def test_interrupted_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "priors.json"
    store = StructurePriorStore(path)
    _prior_update(store, "A")
    before = path.read_bytes()

    def disk_full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("services.durable_json.os.replace", disk_full)
    with pytest.raises(PersistenceError):
        _prior_update(store, "B")

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
    # Memory must not claim what never reached disk.
    assert [o["observation_id"] for o in store._data["otm_strangle"]["observations"]] == ["A"]


def test_interrupted_calibration_write_leaves_file_and_memory_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    store = IVExpansionCalibration(store_path=path)
    store.update(0.5, 5.0, observation_id="A", observation_date=DAY)
    before = path.read_bytes()

    def disk_full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("services.durable_json.os.replace", disk_full)
    with pytest.raises(PersistenceError):
        store.update(0.6, 6.0, observation_id="B", observation_date=DAY)

    assert path.read_bytes() == before
    assert store._n() == 1
    assert store._observation_ids == {"A"}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_non_finite_and_boolean_values_are_rejected(tmp_path, bad):
    cal = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    prior = StructurePriorStore(tmp_path / "priors.json")

    with pytest.raises(ValueError):
        cal.update(0.5, bad, observation_id="Z", observation_date=DAY)
    with pytest.raises(ValueError):
        _prior_update(prior, "Z", ret=bad)

    assert not (tmp_path / "cal.json").exists()
    assert not (tmp_path / "priors.json").exists()


def test_finalize_marks_a_persistence_failure_as_retryable(tmp_path, monkeypatch):
    import services.calibration_service as calibration_service
    import services.structure_prior_store as structure_prior_store

    broken_cal = IVExpansionCalibration(store_path=_unwritable(tmp_path, "cal.json"))
    priors = StructurePriorStore(tmp_path / "priors.json")
    monkeypatch.setattr(calibration_service, "get_calibration", lambda: broken_cal)
    monkeypatch.setattr(structure_prior_store, "get_structure_prior_store", lambda: priors)
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_trade(store, "T1")

    result = finalize_trade_and_update_learning(
        trade_id="T1", exit_date=DAY, realized_return_pct=5.0, realized_expansion_pct=5.0, store=store,
    )

    assert result["learning_update_status"] == "calibration_failed"
    assert any("calibration update failed" in w for w in result["warnings"])
    assert [row["trade_id"] for row in store.list_trades_with_failed_learning_update()] == ["T1"]
    assert broken_cal._n() == 0


# ── 3. exclusive finalization ─────────────────────────────────────────────────


def _open_trade(store: OutcomeStore, trade_id: str) -> None:
    store.insert_entry(
        trade_id=trade_id, symbol=trade_id, structure="otm_strangle",
        entry_date=date(2026, 4, 25), earnings_date=date(2026, 5, 2),
        setup_score=0.6, source_type="paper", entry_mid=2.0,
    )


def _exit(store: OutcomeStore, trade_id: str, ret: float, owner=None) -> bool:
    return store.update_exit(trade_id=trade_id, exit_date=DAY, exit_mid=2.0,
                             realized_return_pct=ret, realized_expansion_pct=ret, owner=owner)


def test_a_live_claim_is_exclusive(tmp_path):
    db = tmp_path / "o.sqlite"
    worker_a, worker_b = OutcomeStore(db), OutcomeStore(db)  # separate connections
    _open_trade(worker_a, "T1")

    assert worker_a.claim_for_finalization("T1", owner="A") is True
    assert worker_b.claim_for_finalization("T1", owner="B") is False
    # B cannot overwrite A's exit facts or finalize A's claim.
    assert _exit(worker_b, "T1", 99.0) is False
    assert _exit(worker_b, "T1", 99.0, owner="B") is False
    assert worker_b.mark_finalized("T1") is False
    assert worker_b.mark_finalized("T1", owner="B") is False
    # A can, and the row stays claimed until A finalizes it.
    assert _exit(worker_a, "T1", 5.0, owner="A") is True
    assert worker_a.get_trade("T1")["status"] == "finalizing"
    assert worker_a.mark_finalized("T1", owner="A") is True
    row = worker_b.get_trade("T1")
    assert (row["status"], row["realized_return_pct"], row["finalizing_owner"]) == ("finalized", 5.0, None)


def test_a_claim_requires_an_owner(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_trade(store, "T1")
    with pytest.raises(ValueError, match="owner"):
        store.claim_for_finalization("T1", owner="")


def test_an_expired_claim_can_be_taken_over(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_trade(store, "T1")
    assert store.claim_for_finalization("T1", owner="crashed") is True
    store._conn.execute(
        "UPDATE outcome_trades SET finalizing_since = datetime('now', '-2 hours') WHERE trade_id = 'T1'"
    )
    store._conn.commit()

    assert store.claim_for_finalization("T1", owner="recovery") is True
    assert store.get_trade("T1")["finalizing_owner"] == "recovery"
    # The crashed worker can no longer write.
    assert _exit(store, "T1", 1.0, owner="crashed") is False


def test_invalidation_is_refused_during_a_live_claim_but_allowed_after_expiry(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_trade(store, "T1")
    store.claim_for_finalization("T1", owner="A")

    with pytest.raises(ValueError, match="being finalized"):
        store.invalidate("T1", reason="late")
    store._conn.execute(
        "UPDATE outcome_trades SET finalizing_since = datetime('now', '-2 hours') WHERE trade_id = 'T1'"
    )
    store._conn.commit()
    assert store.invalidate("T1", reason="late")["invalidation_reason"] == "late"


def test_second_finalizer_is_refused_while_the_first_holds_the_claim(tmp_path, monkeypatch):
    db = tmp_path / "o.sqlite"
    worker_a, worker_b = OutcomeStore(db), OutcomeStore(db)
    _open_trade(worker_a, "T1")
    learned = []
    import services.calibration_service as calibration_service
    monkeypatch.setattr(calibration_service, "get_calibration", lambda: learned.append(1) or pytest.fail("cal"))

    assert worker_a.claim_for_finalization("T1", owner="A") is True
    with pytest.raises(ValueError, match="another worker"):
        finalize_trade_and_update_learning(
            trade_id="T1", exit_date=DAY, realized_return_pct=-50.0, realized_expansion_pct=-50.0,
            store=worker_b,
        )

    assert learned == []
    assert worker_b.get_trade("T1")["realized_return_pct"] is None


def test_refinalizing_a_finalized_trade_reruns_learning_idempotently(tmp_path, monkeypatch):
    import services.calibration_service as calibration_service
    import services.structure_prior_store as structure_prior_store

    cal = IVExpansionCalibration(store_path=tmp_path / "cal.json")
    priors = StructurePriorStore(tmp_path / "priors.json")
    monkeypatch.setattr(calibration_service, "get_calibration", lambda: cal)
    monkeypatch.setattr(structure_prior_store, "get_structure_prior_store", lambda: priors)
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_trade(store, "T1")
    kwargs = dict(trade_id="T1", exit_date=DAY, realized_return_pct=5.0, realized_expansion_pct=5.0, store=store)

    first = finalize_trade_and_update_learning(**kwargs)
    second = finalize_trade_and_update_learning(**{**kwargs, "realized_return_pct": 77.0})

    assert first["learning_update_status"] == second["learning_update_status"] == "complete"
    assert _cal_ids(tmp_path / "cal.json") == ["T1"]
    assert store.get_trade("T1")["realized_return_pct"] == 5.0  # finalized facts are not rewritten


def _finalize_worker(db: str, ret: float, barrier, results) -> None:
    # Store paths come from the environment inherited from the parent (set
    # before spawn), since the modules resolve them at import time.
    from services.outcome_recorder import OutcomeStore as Store
    from services.outcome_recorder import finalize_trade_and_update_learning as finalize

    store = Store(Path(db))
    barrier.wait()
    try:
        finalize(trade_id="RACE", exit_date=DAY, realized_return_pct=ret, realized_expansion_pct=ret, store=store)
        results.put(("ok", ret))
    except ValueError:
        results.put(("refused", ret))


def test_concurrent_finalizers_in_separate_processes_finalize_once(tmp_path, monkeypatch):
    db, cal, prior = tmp_path / "o.sqlite", tmp_path / "cal.json", tmp_path / "priors.json"
    monkeypatch.setenv("OPTIONS_CALCULATOR_CALIBRATION_PATH", str(cal))
    monkeypatch.setenv("OPTIONS_CALCULATOR_PRIORS_PATH", str(prior))
    _open_trade(OutcomeStore(db), "RACE")
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(2), ctx.Queue()
    procs = [
        ctx.Process(target=_finalize_worker, args=(str(db), ret, barrier, results))
        for ret in (10.0, -10.0)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
        assert proc.exitcode == 0
    outcomes = sorted(results.get(timeout=5) for _ in procs)

    winners = [ret for status, ret in outcomes if status == "ok"]
    # Exactly one worker finalized; the other was refused or re-ran learning
    # idempotently on the already-finalized trade.
    row = OutcomeStore(db).get_trade("RACE")
    assert row["status"] == "finalized"
    assert row["realized_return_pct"] in winners
    assert _cal_ids(cal) == ["RACE"]
    assert _prior_ids(prior) == ["RACE"]
