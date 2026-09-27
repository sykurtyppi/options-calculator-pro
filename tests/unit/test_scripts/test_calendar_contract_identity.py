"""Calendar spreads must be entered and exited on one strike, the booked one.

Calendars are the primary research hypothesis, and before this fix their
pricing chose each leg by nearest strike: discovery could book a diagonal
(back leg at a different strike) and a reprice could value different
contracts than the ones held, with no exit verification at all.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_forward_loop as forward_loop
from services.baseline_evidence_store import BaselineEvidenceStore
from services.outcome_recorder import OutcomeStore

FRONT = "2026-05-01"
BACK = "2026-05-22"
EARNINGS = date(2026, 4, 28)
T_MINUS_1 = EARNINGS - timedelta(days=1)
BOOKED = {"strike": 100.0, "calendar_back_strike": 100.0, "front_expiry": FRONT, "back_expiry": BACK}


def _frame(strikes: list[float], *, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"contractSymbol": f"{prefix}{strike:g}", "strike": strike, "bid": 1.0 + i, "ask": 1.2 + i, "lastPrice": 1.1 + i}
            for i, strike in enumerate(strikes)
        ],
        columns=["contractSymbol", "strike", "bid", "ask", "lastPrice"],
    )


def _install_chains(monkeypatch, *, front: list[float], back: list[float], spot: float = 100.0) -> None:
    chains = {
        FRONT: SimpleNamespace(calls=_frame(front, prefix="FC"), puts=_frame(front, prefix="FP")),
        BACK: SimpleNamespace(calls=_frame(back, prefix="BC"), puts=_frame(back, prefix="BP")),
    }

    class _Ticker:
        options = [FRONT, BACK]

        def option_chain(self, expiry: str) -> SimpleNamespace:
            return chains[expiry]

    monkeypatch.setattr(forward_loop.yf, "Ticker", lambda _symbol: _Ticker())
    monkeypatch.setattr(forward_loop, "_latest_spot_price", lambda _symbol: spot)
    monkeypatch.setattr(forward_loop, "record_provider_telemetry", lambda **_kwargs: None)


def _quote(structure="call_calendar", context=None, as_of=date(2026, 4, 21)):
    return forward_loop.fetch_structure_quote(
        symbol="AAPL", structure=structure, earnings_date=EARNINGS, as_of_date=as_of, context=context,
    )


# ── pricing ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("structure", ["call_calendar", "put_calendar"])
def test_discovery_books_both_legs_at_one_strike(monkeypatch, structure):
    _install_chains(monkeypatch, front=[95.0, 100.0, 105.0], back=[95.0, 100.0, 105.0], spot=100.4)

    quote = _quote(structure)

    context = quote["context"]
    assert context["strike"] == context["calendar_back_strike"] == 100.0
    assert (context["front_expiry"], context["back_expiry"]) == (FRONT, BACK)
    assert forward_loop._entry_context_verifiable(structure, context)


def test_discovery_refuses_a_diagonal(monkeypatch):
    # The back month lacks the front strike; nearest-row used to book 101.
    _install_chains(monkeypatch, front=[100.0], back=[101.0, 105.0])

    quote = _quote()

    assert quote["mid"] is None
    assert quote["reason"] == "missing_back_leg"


@pytest.mark.parametrize(
    ("front", "back"),
    [([95.0, 105.0], [100.0]), ([100.0], [95.0, 105.0])],
    ids=["front_strike_gone", "back_strike_gone"],
)
def test_reprice_requires_the_booked_strike_on_both_legs(monkeypatch, front, back):
    _install_chains(monkeypatch, front=front, back=back)

    quote = _quote(context=dict(BOOKED), as_of=T_MINUS_1)

    assert quote["mid"] is None
    assert quote["reason"] == "booked_contract_unavailable"


def test_reprice_of_booked_calendar_prices_the_same_contracts(monkeypatch):
    _install_chains(monkeypatch, front=[95.0, 100.0], back=[100.0, 110.0], spot=108.0)

    quote = _quote(context=dict(BOOKED), as_of=T_MINUS_1)

    assert quote["context"]["strike"] == quote["context"]["calendar_back_strike"] == 100.0
    assert forward_loop._booked_contracts_match("call_calendar", BOOKED, quote["context"])


def test_legacy_reprice_without_back_strike_uses_the_front_strike(monkeypatch):
    _install_chains(monkeypatch, front=[100.0], back=[100.0, 101.0])
    legacy = {"strike": 100.0, "front_expiry": FRONT, "back_expiry": BACK}

    quote = _quote(context=legacy, as_of=T_MINUS_1)

    assert quote["context"]["calendar_back_strike"] == 100.0
    assert not forward_loop._entry_context_verifiable("call_calendar", legacy)


# ── contract check ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exit_context",
    [
        {**BOOKED, "calendar_back_strike": 105.0},
        {**BOOKED, "back_expiry": "2026-05-29"},
        {**BOOKED, "front_expiry": "2026-05-08"},
        {k: v for k, v in BOOKED.items() if k != "back_expiry"},
    ],
    ids=["back_strike", "back_expiry", "front_expiry", "missing_back_expiry"],
)
def test_calendar_exit_on_any_other_leg_or_expiry_is_rejected(exit_context):
    assert forward_loop._booked_contracts_match("put_calendar", BOOKED, BOOKED)
    assert not forward_loop._booked_contracts_match("put_calendar", BOOKED, exit_context)


def test_calendar_entry_without_back_expiry_is_unverifiable():
    entry = {k: v for k, v in BOOKED.items() if k != "back_expiry"}
    assert not forward_loop._entry_context_verifiable("call_calendar", entry)


def test_symbol_conflict_is_detected_even_when_strikes_cannot_be_verified():
    legacy = {"strike": 100.0, "front_contract": "FC100", "back_contract": "BC101"}
    exit_context = {**BOOKED, "front_contract": "FC100", "back_contract": "BC100"}

    assert forward_loop._contract_symbols_conflict("call_calendar", legacy, exit_context)
    assert not forward_loop._contract_symbols_conflict("call_calendar", legacy, {**exit_context, "back_contract": "BC101"})


# ── selector exit wiring ──────────────────────────────────────────────────────


def _open_calendar(store, trade_id, context):
    store.insert_entry(
        trade_id=trade_id, symbol=trade_id, structure="call_calendar",
        entry_date=EARNINGS - timedelta(days=6), earnings_date=EARNINGS,
        setup_score=0.6, source_type="paper", entry_mid=1.5, execution_penalty_at_entry=0.0,
        notes=json.dumps({"pricing_context": context}),
    )


def _exit(tmp_path, store, exit_context, captured):
    return forward_loop.run_exit_detection(
        today=T_MINUS_1, store=store, log_path=tmp_path / "log.jsonl",
        price_fetcher=lambda **_: {"mid": 2.0, "context": exit_context},
        finalizer=lambda **kw: captured.append(kw) or {},
        baseline_store=BaselineEvidenceStore(tmp_path / "b.sqlite"),
    )


def test_selector_calendar_exit_is_verified(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_calendar(store, "CRM", BOOKED)
    captured = []

    _exit(tmp_path, store, dict(BOOKED), captured)

    assert captured[0]["exit_execution_scenarios"]["contract_verification"] == "verified"


def test_selector_calendar_exit_on_other_back_expiry_fails_closed(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    _open_calendar(store, "CRM", BOOKED)
    captured = []

    summary = _exit(tmp_path, store, {**BOOKED, "back_expiry": "2026-05-29"}, captured)

    assert captured == []
    assert summary["skipped"] == 1
    assert store.get_trade("CRM")["last_exit_attempt_reason"] == "booked_contract_mismatch"


def test_legacy_selector_calendar_with_conflicting_symbol_fails_closed(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    legacy = {"strike": 100.0, "front_expiry": FRONT, "back_expiry": BACK,
              "front_contract": "FC100", "back_contract": "BC101"}
    _open_calendar(store, "CRM", legacy)
    captured = []

    _exit(tmp_path, store, {**BOOKED, "front_contract": "FC100", "back_contract": "BC100"}, captured)

    assert captured == []
    assert store.get_trade("CRM")["last_exit_attempt_reason"] == "booked_contract_mismatch"


def test_legacy_selector_calendar_without_conflict_proceeds_flagged(tmp_path):
    store = OutcomeStore(tmp_path / "o.sqlite")
    legacy = {"strike": 100.0, "front_expiry": FRONT, "back_expiry": BACK}
    _open_calendar(store, "CRM", legacy)
    captured = []

    _exit(tmp_path, store, dict(BOOKED), captured)

    assert captured[0]["exit_execution_scenarios"]["contract_verification"] == "unverifiable_entry_context"


def test_reprice_uses_the_stored_back_strike_for_the_back_leg(monkeypatch):
    # New entries always store equal strikes; the back leg must still follow
    # its own recorded strike rather than silently assume the front one.
    _install_chains(monkeypatch, front=[100.0], back=[100.0, 101.0])
    booked = {**BOOKED, "calendar_back_strike": 101.0}

    quote = _quote(context=booked, as_of=T_MINUS_1)

    assert quote["context"]["calendar_back_strike"] == 101.0
    assert quote["context"]["back_contract"] == "BC101"
