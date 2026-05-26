from __future__ import annotations

import pytest

from trading_system.database import get_conn, init_db, insert_trade


def test_get_conn_uses_busy_timeout_and_db_is_readable(tmp_path) -> None:
    db_path = tmp_path / "trading_system.db"

    init_db(db_path)

    with get_conn(db_path) as conn:
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]

    assert busy_timeout >= 5000
    assert quick_check == "ok"


def test_dynamic_insert_rejects_unknown_trade_columns(tmp_path) -> None:
    db_path = tmp_path / "trading_system.db"
    init_db(db_path)

    with pytest.raises(ValueError, match="Invalid trades columns"):
        insert_trade(
            {
                "symbol": "AAPL",
                "event_date": "2026-05-01",
                "status": "open",
                "bad_column) VALUES ('x'); --": "boom",
            },
            db_path=db_path,
        )
