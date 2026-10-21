"""The research database closes every connection it opens.

Its connections used to stay open until a cycle collection. A process that
forked while holding them handed each child inherited connections; a child
closing one at an arbitrary moment could checkpoint away or truncate the
shared WAL under processes still using it. Several processes opening an
older store at once then lost the migration's committed column or failed
with SQLITE_IOERR_SHORT_READ (test_store_migrations, research_db).
"""
from __future__ import annotations

import gc
import sqlite3
from pathlib import Path

import pytest

import services.sqlite_helpers as sqlite_helpers
from services.institutional_ml_db import InstitutionalMLDatabase


@pytest.fixture
def opened(monkeypatch):
    """Every connection the store opens, recorded as it is opened."""
    conns: list[sqlite3.Connection] = []
    real = sqlite_helpers.open_db_conn

    def recording(*args, **kwargs):
        conn = real(*args, **kwargs)
        conns.append(conn)
        return conn

    monkeypatch.setattr(sqlite_helpers, "open_db_conn", recording)
    return conns


def _is_closed(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def test_every_connection_is_closed_when_the_call_returns(tmp_path: Path, opened) -> None:
    gc.disable()  # closing must not depend on a collection happening
    try:
        db = InstitutionalMLDatabase(db_path=str(tmp_path / "r.db"))
        db.get_backtest_results()
        db.get_backtest_trades("no-such-session")
        assert len(opened) >= 4
        assert [_is_closed(c) for c in opened] == [True] * len(opened)
    finally:
        gc.enable()


def test_connection_commits_on_success_and_rolls_back_on_error(tmp_path: Path, opened) -> None:
    path = tmp_path / "r.db"
    db = InstitutionalMLDatabase(db_path=str(path))
    with db._connection() as conn:
        conn.execute("DELETE FROM ticker_universe WHERE symbol = 'AAPL'")
    with pytest.raises(RuntimeError):
        with db._connection() as conn:
            conn.execute("DELETE FROM ticker_universe WHERE symbol = 'MSFT'")
            raise RuntimeError("boom")
    assert all(_is_closed(c) for c in opened)
    check = sqlite3.connect(path)
    symbols = {row[0] for row in check.execute("SELECT symbol FROM ticker_universe")}
    check.close()
    assert "AAPL" not in symbols and "MSFT" in symbols


def test_no_connection_is_opened_outside_the_closing_context() -> None:
    source = Path(InstitutionalMLDatabase.__module__.replace(".", "/") + ".py")
    text = (Path(__file__).resolve().parents[3] / source).read_text()
    # The one call is inside _connection itself.
    assert text.count("self._open_conn()") == 1
