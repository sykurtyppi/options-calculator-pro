"""Codex F1: the persisted historical_vs_implied_move_ratio changed scale.
Rows must carry which scale they use; the migration must add the column to
pre-existing stores; legacy rows stay NULL. The insert here is a REAL insert,
so SQLite itself enforces that the column list, placeholders and params agree.
"""
from __future__ import annotations

import inspect
import sqlite3
from datetime import date
from pathlib import Path

from services.move_statistics import MOVE_RATIO_UNITS_VERSION
from services.outcome_recorder import OutcomeStore, _open_db


def _required_kwargs(fn):
    dummies = {str: "x", date: date(2026, 1, 15), float: 1.0, int: 1, bool: True}
    out = {}
    for name, prm in inspect.signature(fn).parameters.items():
        if name == "self" or prm.default is not inspect.Parameter.empty or prm.kind is prm.VAR_KEYWORD:
            continue
        ann = prm.annotation
        for typ, val in dummies.items():
            if ann is typ or (isinstance(ann, str) and ann == typ.__name__):
                out[name] = val
                break
        else:
            out[name] = "x"
    return out


def _columns(path: Path) -> set[str]:
    con = sqlite3.connect(path)
    try:
        return {r[1] for r in con.execute("PRAGMA table_info(outcome_trades)")}
    finally:
        con.close()


def test_fresh_store_has_units_column(tmp_path: Path):
    p = tmp_path / "o.sqlite"
    _open_db(p).close()
    assert "move_ratio_units_version" in _columns(p)


def test_migration_adds_column_to_pre_f1_store(tmp_path: Path):
    p = tmp_path / "o.sqlite"
    _open_db(p).close()
    con = sqlite3.connect(p)
    con.execute("ALTER TABLE outcome_trades DROP COLUMN move_ratio_units_version")
    con.commit()
    con.close()
    assert "move_ratio_units_version" not in _columns(p)
    _open_db(p).close()  # additive migration re-adds it
    assert "move_ratio_units_version" in _columns(p)


def test_units_version_is_persisted_and_legacy_rows_stay_null(tmp_path: Path):
    p = tmp_path / "o.sqlite"
    store = OutcomeStore(store_path=p)
    base = _required_kwargs(OutcomeStore.insert_entry)
    base.update(symbol="AAPL", structure="atm_straddle")
    store.insert_entry(**{**base, "trade_id": "t-new", "historical_vs_implied_move_ratio": 1.0,
                          "move_ratio_units_version": MOVE_RATIO_UNITS_VERSION})
    store.insert_entry(**{**base, "trade_id": "t-legacy", "historical_vs_implied_move_ratio": 0.755})
    con = sqlite3.connect(p)
    rows = dict(con.execute(
        "SELECT trade_id, move_ratio_units_version FROM outcome_trades WHERE trade_id IN ('t-new','t-legacy')"
    ).fetchall())
    con.close()
    assert rows["t-new"] == 2
    assert rows["t-legacy"] is None
