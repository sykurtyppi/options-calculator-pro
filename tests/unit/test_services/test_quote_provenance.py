"""Where and when an analysis's quotes came from, and the forward loop's refusal of bad ones.

Every forward analysis reached the ledger with quote_source and
quote_timestamp NULL, so stops caused by after-hours yfinance quotes could not
be told apart from stops caused by the market.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_forward_loop as forward_loop
from services import evidence_funnel as ef
from services.baseline_evidence_store import BaselineEvidenceStore
from services.market_calendar import in_regular_session
from services.outcome_recorder import OutcomeStore
from services.quote_provenance import (
    BASIS_FETCH,
    BASIS_PROVIDER,
    PROBLEM_MISSING,
    PROBLEM_OUTSIDE_SESSION,
    chain_quote_provenance,
    ledger_quote_payload,
    quote_provenance_problem,
)
from services.recommendation_ledger import RecommendationLedger, build_record_from_analysis
from tests.unit.quote_fixtures import IN_SESSION_QUOTES

IN_SESSION = datetime(2026, 11, 4, 16, 45, tzinfo=timezone.utc)   # Wednesday 11:45 EST
AFTER_CLOSE = datetime(2026, 11, 4, 21, 30, tzinfo=timezone.utc)  # 16:30 EST


def _ts(moment: datetime) -> float:
    return moment.timestamp()


# ── provenance ──────────────────────────────────────────────────────────────


def test_a_provider_chain_records_the_providers_quote_time() -> None:
    frame = pd.DataFrame({"updated": [_ts(IN_SESSION - timedelta(minutes=m)) for m in (1, 3, 9)]})
    p = chain_quote_provenance(frame, provider="marketdata", fetched_at=IN_SESSION)
    assert p["quote_source"] == "marketdata" and p["timestamp_basis"] == BASIS_PROVIDER
    assert p["quote_timestamp"] == (IN_SESSION - timedelta(minutes=3)).isoformat()  # the median
    assert p["oldest_quote_timestamp"] == (IN_SESSION - timedelta(minutes=9)).isoformat()
    assert p["quote_age_minutes"] == 3.0 and p["contracts_with_quote_time"] == 3
    assert p["in_regular_session"] is True and p["quote_session"] == "2026-11-04"


def test_an_even_count_takes_the_older_middle_quote() -> None:
    frame = pd.DataFrame({"updated": [_ts(IN_SESSION - timedelta(minutes=m)) for m in (2, 4, 6, 8)]})
    p = chain_quote_provenance(frame, provider="marketdata", fetched_at=IN_SESSION)
    assert p["quote_age_minutes"] == 6.0


def test_yfinance_records_the_fetch_time_and_never_invents_a_quote_time() -> None:
    frame = pd.DataFrame({"bid": [1.0], "ask": [1.1]})
    p = chain_quote_provenance(frame, provider="yfinance", fetched_at=AFTER_CLOSE)
    assert p["timestamp_basis"] == BASIS_FETCH and p["quote_timestamp"] == AFTER_CLOSE.isoformat()
    assert p["quote_age_minutes"] is None and p["oldest_quote_timestamp"] is None
    assert p["in_regular_session"] is False


def test_before_the_open_the_quotes_are_the_previous_sessions() -> None:
    pre_open = datetime(2026, 11, 4, 13, 0, tzinfo=timezone.utc)  # 08:00 EST
    p = chain_quote_provenance(pd.DataFrame(), provider="yfinance", fetched_at=pre_open)
    assert p["in_regular_session"] is False and p["quote_session"] == "2026-11-03"


def test_unusable_provider_times_are_ignored() -> None:
    frame = pd.DataFrame({"updated": [None, 0, float("nan"), "x", _ts(IN_SESSION - timedelta(minutes=5))]})
    p = chain_quote_provenance(frame, provider="marketdata", fetched_at=IN_SESSION)
    assert p["contracts_with_quote_time"] == 1 and p["quote_age_minutes"] == 5.0
    assert chain_quote_provenance(None, provider=None, fetched_at=IN_SESSION)["timestamp_basis"] == BASIS_FETCH


def test_a_naive_fetch_time_is_refused() -> None:
    with pytest.raises(ValueError):
        chain_quote_provenance(pd.DataFrame(), provider="yfinance", fetched_at=datetime(2026, 11, 4, 12, 0))


@pytest.mark.parametrize("provenance,problem", [
    (None, PROBLEM_MISSING),
    ({}, PROBLEM_MISSING),
    ({**IN_SESSION_QUOTES, "in_regular_session": False}, PROBLEM_OUTSIDE_SESSION),
    ({**IN_SESSION_QUOTES, "in_regular_session": None}, PROBLEM_OUTSIDE_SESSION),
    ({**IN_SESSION_QUOTES, "quote_age_minutes": 45.4}, "stale_quotes:45m"),
    ({**IN_SESSION_QUOTES, "quote_age_minutes": 30.0}, None),
    ({**IN_SESSION_QUOTES, "quote_age_minutes": None}, None),  # yfinance: no provider time
    (IN_SESSION_QUOTES, None),
])
def test_quote_provenance_problem(provenance, problem) -> None:
    assert quote_provenance_problem(provenance) == problem


# ── the ledger ──────────────────────────────────────────────────────────────


def _analysis(metrics: dict) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="AAPL", recommendation="No Trade", confidence_pct=0.0, setup_score=0.0, metrics=metrics,
        rationale=[], selector_output={"recommendation": "No Trade", "earnings_date": "2026-11-10"},
        structure_scorecards=[], vol_snapshot={"symbol": "AAPL", "as_of_date": "2026-11-04",
                                               "earnings_date": "2026-11-10"},
    )


def test_the_ledger_quote_columns_come_from_the_analysis_provenance() -> None:
    record = build_record_from_analysis(_analysis({"quote_provenance": IN_SESSION_QUOTES}))
    assert (record.quote_source, record.quote_timestamp, record.quote_quality) == \
        ("marketdata", "2026-11-04T16:44:00+00:00", "provider_quote_time")
    assert record.metadata["quote_provenance"] == IN_SESSION_QUOTES
    assert ledger_quote_payload(None) == {}


def test_an_explicit_quote_payload_still_wins() -> None:
    record = build_record_from_analysis(_analysis({"quote_provenance": IN_SESSION_QUOTES}),
                                        quote_payload={"quote_source": "manual", "quote_timestamp": "t"})
    assert (record.quote_source, record.quote_timestamp) == ("manual", "t")


def test_the_analyser_stamps_every_analysis() -> None:
    from tests.unit.test_web.test_analyze_single_ticker_golden import _run_watch_scenario

    before = datetime.now(timezone.utc)
    provenance = _run_watch_scenario().metrics["quote_provenance"]
    fetched = datetime.fromisoformat(provenance["fetched_at"])
    assert before <= fetched <= datetime.now(timezone.utc)
    # The scenario has no provider client, so the chain is the yfinance fallback.
    assert provenance["quote_source"] == "yfinance" and provenance["timestamp_basis"] == BASIS_FETCH
    assert provenance["in_regular_session"] is in_regular_session(fetched)


# ── the forward loop ────────────────────────────────────────────────────────

TODAY = date(2026, 11, 4)


def _candidate(provenance) -> SimpleNamespace:
    earnings = TODAY + timedelta(days=6)
    metrics = {} if provenance is None else {"quote_provenance": provenance}
    return SimpleNamespace(
        symbol="GOOD", recommendation="Candidate", confidence_pct=70.0, setup_score=0.6, metrics=metrics,
        rationale=["ok"],
        selector_output={"recommendation": "Candidate", "best_structure": "atm_straddle",
                         "earnings_date": earnings.isoformat(), "binding_gate": None},
        structure_scorecards=[{"structure": "atm_straddle", "execution_penalty": 0.03}],
        vol_snapshot={"symbol": "GOOD", "as_of_date": TODAY.isoformat(), "earnings_date": earnings.isoformat(),
                      "days_to_earnings": 6, "data_quality_score": 0.9},
    )


def _run(tmp_path, provenance):
    funnel = ef.EvidenceFunnelStore(tmp_path / "funnel.sqlite")
    ledger = RecommendationLedger(ledger_path=tmp_path / "r.sqlite")
    baselines = BaselineEvidenceStore(tmp_path / "b.sqlite")
    priced: list[str] = []

    def price_fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):  # noqa: ARG001
        priced.append(structure)
        return {"mid": 5.0, "context": {}}

    summary = forward_loop.run_forward_screener(
        today=TODAY, store=OutcomeStore(store_path=tmp_path / "o.sqlite"), ledger=ledger,
        baseline_store=baselines, log_path=tmp_path / "log.jsonl",
        screener_builder=lambda **_: {"rows": [{"symbol": "GOOD", "status": "ranked", "dte": 6}]},
        analyzer=lambda symbol, mda_client=None: _candidate(provenance), price_fetcher=price_fetcher,
        funnel=funnel, cycle_run_id="cycle-1", symbols=["GOOD"],
    )
    eligible = [e for e in funnel.events("cycle-1") if e["stage"] == ef.STAGE_ELIGIBLE]
    rows = ledger._conn.execute("SELECT quote_source, quote_timestamp, quote_quality FROM recommendations").fetchall()
    funnel.close()
    return summary, eligible, priced, [tuple(r) for r in rows]


def test_in_session_quotes_enter_as_before(tmp_path) -> None:
    summary, eligible, priced, rows = _run(tmp_path, IN_SESSION_QUOTES)
    assert summary["entries"] == 1 and eligible[0]["outcome"] == ef.OUTCOME_ADVANCED
    assert rows == [("marketdata", "2026-11-04T16:44:00+00:00", "provider_quote_time")]


@pytest.mark.parametrize("provenance,reason", [
    ({**IN_SESSION_QUOTES, "in_regular_session": False}, "untrusted_quotes:outside_session"),
    ({**IN_SESSION_QUOTES, "quote_age_minutes": 90.0}, "untrusted_quotes:stale_quotes:90m"),
    (None, "untrusted_quotes:missing_quote_provenance"),
])
def test_untrusted_quotes_price_neither_the_trade_nor_its_shadow(tmp_path, provenance, reason) -> None:
    summary, eligible, priced, rows = _run(tmp_path, provenance)
    assert summary["entries"] == 0 and summary["skip_reasons"].get("untrusted_quotes") == 1
    assert [(e["outcome"], e["reason"]) for e in eligible] == [(ef.OUTCOME_DROPPED, reason)]
    assert priced == []  # no entry quote, no universe shadow quote
    assert len(rows) == 1  # the analysis itself is still on record
