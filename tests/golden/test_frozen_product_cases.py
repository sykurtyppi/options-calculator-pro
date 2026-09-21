"""Golden product cases using only frozen point-in-time inputs."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import services.structure_prior_store as prior_store_module
from services.earnings_vol_snapshot import build_vol_snapshot
from services.structure_prior_store import StructurePriorStore
from services.structure_scorecard import build_structure_scorecards
from services.structure_selector import select_best_structure

CASE = json.loads((Path(__file__).parent / "frozen_earnings_chain.json").read_text())


def _snapshot(chain):
    prices = pd.DataFrame(CASE["price_history"])
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    options = pd.DataFrame(chain)
    return build_vol_snapshot(
        CASE["symbol"], date.fromisoformat(CASE["as_of_date"]),
        option_chain_data=options, earnings_metadata=CASE["earnings"], price_data=prices,
    )


def _decision(snapshot, monkeypatch, tmp_path):
    store = StructurePriorStore(tmp_path / "isolated-priors.json")
    monkeypatch.setattr(prior_store_module, "get_structure_prior_store", lambda: store)
    cards = build_structure_scorecards(snapshot, as_of_date=snapshot.as_of_date)
    output = select_best_structure(snapshot, cards)
    ranked = sorted(
        (card for card in cards if card.eligible),
        key=lambda card: (card.composite_structure_score, card.expected_edge_pct),
        reverse=True,
    )
    return output, ranked


def test_frozen_earnings_chain_matches_expected_decision_inputs(monkeypatch, tmp_path):
    snapshot = _snapshot(CASE["option_chain"])
    actual = snapshot.to_dict()
    for field, expected in CASE["expected"].items():
        if field in {"lead_structure", "recommendation"}:
            continue
        if isinstance(expected, float):
            assert actual[field] == pytest.approx(expected, rel=1e-12, abs=1e-12)
        else:
            assert actual[field] == expected

    # Independent oracle: nearest-expiry ATM call + put midpoint divided by the
    # contemporaneous underlying. This is intentionally not production code.
    chain = CASE["option_chain"]
    near_expiry = min(row["expiry"] for row in chain)
    near = [row for row in chain if row["expiry"] == near_expiry]
    spot = CASE["expected"]["underlying_price"]
    atm_strike = min({row["strike"] for row in near}, key=lambda strike: abs(strike - spot))
    atm = [row for row in near if row["strike"] == atm_strike]
    call_mid = next(row["mid"] for row in atm if row["call_put"] == "C")
    put_mid = next(row["mid"] for row in atm if row["call_put"] == "P")
    oracle_implied_move = 100.0 * (call_mid + put_mid) / spot
    assert snapshot.near_term_implied_move_pct == pytest.approx(oracle_implied_move)

    decision, ranked = _decision(snapshot, monkeypatch, tmp_path)
    assert ranked[0].structure == CASE["expected"]["lead_structure"]
    assert decision.recommendation == CASE["expected"]["recommendation"]


def test_future_chain_rows_cannot_change_point_in_time_result():
    baseline = _snapshot(CASE["option_chain"]).to_dict()
    poison = dict(CASE["option_chain"][0])
    poison.update({"trade_date": "2026-04-21", "bid": 999.0, "ask": 1000.0, "impliedVolatility": 9.0})
    actual = _snapshot([*CASE["option_chain"], poison]).to_dict()
    assert actual == baseline


def test_provider_outage_degrades_to_real_selector_no_trade(monkeypatch, tmp_path):
    snapshot = _snapshot(CASE["provider_outage"]["option_chain"])
    actual = snapshot.to_dict()
    expected = CASE["provider_outage"]["expected"]
    assert actual["iv30"] is expected["iv30"]
    assert actual["near_term_implied_move_pct"] is expected["near_term_implied_move_pct"]
    assert actual["surface_quality_status"] == expected["surface_quality_status"]
    assert actual["surface_quality_reasons"] == expected["surface_quality_reasons"]

    decision, ranked = _decision(snapshot, monkeypatch, tmp_path)
    assert ranked == []
    assert decision.recommendation == expected["recommendation"]
    assert decision.best_structure is None
    assert "failed a hard eligibility rule" in decision.why_this_structure[0]
