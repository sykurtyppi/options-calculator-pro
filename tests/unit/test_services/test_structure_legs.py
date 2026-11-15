"""One set of rules for the contracts a structure books, used by scoring and by entry.

The selector scored strangles and condors on the near-term ATM spreads while
the forward loop picked the real OTM legs at entry, on another expiry; and the
loop's expiry rule (first expiry on or after the earnings DATE) could book an
option expiring before an after-close report's reaction.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_forward_loop as forward_loop
from services import structure_legs as sl
from services.structure_scorecard import (
    LEGS_UNAVAILABLE_FLAG,
    score_atm_straddle,
    score_call_calendar,
    score_iron_condor,
    score_otm_strangle,
)
from tests.unit.test_services.test_structure_scorecard import _base_snapshot

STRIKES = [80, 85, 90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110, 115, 120]


def _side(spread: float = 0.10, wide_otm: float | None = None, kind: str = "C") -> pd.DataFrame:
    rows = []
    for strike in STRIKES:
        mid = max(0.5, 10.0 - abs(strike - 100) * 0.3)
        half = (wide_otm if wide_otm is not None and abs(strike - 100) >= 2.5 else spread) / 2
        rows.append({"strike": float(strike), "bid": mid - half, "ask": mid + half,
                     "contractSymbol": f"X{kind}{strike}"})
    return pd.DataFrame(rows)


# ── the trade expiry spans the reaction ─────────────────────────────────────


EXPIRIES = ["2026-11-13", "2026-11-20", "2026-12-04", "2026-12-18"]


@pytest.mark.parametrize("earnings,timing,expiry", [
    (date(2026, 11, 13), "after market close", date(2026, 11, 20)),   # Friday close: same-day expiry is before the reaction
    (date(2026, 11, 13), "before market open", date(2026, 11, 13)),   # BMO: the same-day expiry spans it
    (date(2026, 11, 13), "UNKNOWN", date(2026, 11, 20)),              # unknown is treated as after the close
    (date(2026, 11, 13), None, date(2026, 11, 20)),
    (date(2026, 12, 18), "AMC", None),                                # nothing listed after
])
def test_trade_expiry(earnings, timing, expiry) -> None:
    assert sl.trade_expiry(EXPIRIES, earnings, timing) == expiry


def test_calendar_back_expiry_is_at_least_two_weeks_out() -> None:
    assert sl.calendar_back_expiry(EXPIRIES, date(2026, 11, 20)) == date(2026, 12, 4)
    assert sl.calendar_back_expiry(EXPIRIES, date(2026, 12, 18)) is None


# ── discovery ───────────────────────────────────────────────────────────────


def _strikes(result):
    return {name: leg["strike"] for name, leg in result["legs"].items()}


def test_each_structure_discovers_its_legs() -> None:
    calls, puts = _side(kind="C"), _side(kind="P")
    straddle = sl.discover_legs("atm_straddle", calls, puts, 100.4)
    assert straddle["reason"] is None and _strikes(straddle) == {"call": 100.0, "put": 100.0}
    strangle = sl.discover_legs("otm_strangle", calls, puts, 100.0)
    assert _strikes(strangle) == {"call": 102.5, "put": 97.5}
    condor = sl.discover_legs("iron_condor", calls, puts, 100.0)
    assert _strikes(condor) == {"short_call": 102.5, "long_call": 105.0, "short_put": 97.5, "long_put": 95.0}
    calendar = sl.discover_legs("call_calendar", calls, puts, 100.4, back_calls=calls, back_puts=puts)
    assert _strikes(calendar) == {"front": 100.0, "back": 100.0}
    assert straddle["legs"]["call"]["contract"] == "XC100"


def test_the_leg_spread_is_the_mean_of_the_legs() -> None:
    calls, puts = _side(spread=0.10, wide_otm=1.0, kind="C"), _side(spread=0.10, wide_otm=2.0, kind="P")
    strangle = sl.discover_legs("otm_strangle", calls, puts, 100.0)
    per_leg = [leg["spread_pct"] for leg in strangle["legs"].values()]
    assert len(set(per_leg)) == 2  # different, so the mean is not either leg
    assert strangle["leg_spread_pct"] == pytest.approx(np.mean(per_leg))
    straddle = sl.discover_legs("atm_straddle", calls, puts, 100.0)
    assert strangle["leg_spread_pct"] > straddle["leg_spread_pct"]  # the OTM wings are the wide ones


@pytest.mark.parametrize("structure,mutate,reason", [
    ("atm_straddle", lambda c, p: (c, p[p["strike"] != 100.0]), "missing_atm_pair"),
    ("otm_strangle", lambda c, p: (c[c["strike"] < 100.0], p), "empty_call_chain"),
    ("otm_strangle", lambda c, p: (c[c["strike"] <= 100.0], p), "wing_strike_unavailable"),
    ("iron_condor", lambda c, p: (c[c["strike"] <= 102.5], p), "no_condor_call_wing"),
    ("iron_condor", lambda c, p: (c, p[p["strike"] >= 97.5]), "no_condor_put_wing"),
    ("call_calendar", lambda c, p: (c, p), "no_back_expiry_for_calendar"),
])
def test_an_unformable_structure_says_why(structure, mutate, reason) -> None:
    calls, puts = mutate(_side(kind="C"), _side(kind="P"))
    result = sl.discover_legs(structure, calls, puts, 100.0)
    assert result["reason"] == reason and result["leg_spread_pct"] is None


def test_a_leg_without_a_usable_quote_is_unformable() -> None:
    calls = _side(kind="C")
    calls.loc[calls["strike"] == 100.0, "bid"] = 0.0
    assert sl.discover_legs("atm_straddle", calls, _side(kind="P"), 100.0)["reason"] == "missing_leg_mid"


def test_no_spot_is_unformable() -> None:
    assert sl.discover_legs("atm_straddle", _side(), _side(), float("nan"))["reason"] == "no_spot_price"


# ── entry books what was scored ─────────────────────────────────────────────


def _install(monkeypatch, calls, puts, expiries):
    class _Ticker:
        options = expiries

        def option_chain(self, _expiry):
            return SimpleNamespace(calls=calls, puts=puts)

    monkeypatch.setattr(forward_loop.yf, "Ticker", lambda _s: _Ticker())
    monkeypatch.setattr(forward_loop, "_latest_spot_price", lambda _s: 100.0)
    monkeypatch.setattr(forward_loop, "record_provider_telemetry", lambda **_k: None)


@pytest.mark.parametrize("structure", ["atm_straddle", "otm_strangle", "iron_condor", "call_calendar", "put_calendar"])
def test_entry_discovery_books_the_legs_the_scorer_priced(monkeypatch, structure) -> None:
    calls, puts = _side(kind="C"), _side(kind="P")
    _install(monkeypatch, calls, puts, EXPIRIES)
    quote = forward_loop.fetch_structure_quote(symbol="AAA", structure=structure, earnings_date=date(2026, 11, 13),
                                               as_of_date=date(2026, 11, 10), release_timing="AMC")
    assert quote.get("mid") is not None, quote.get("reason")
    scored = {
        "trade_expiry": "2026-11-20", "back_expiry": "2026-12-04",
        "structures": {structure: sl.discover_legs(structure, calls, puts, 100.0, back_calls=calls, back_puts=puts)},
    }
    check = sl.compare_booked_to_scored(structure, quote["context"], scored)
    assert check["match"] is True, check


def test_entry_never_books_an_expiry_before_an_after_close_reaction(monkeypatch) -> None:
    _install(monkeypatch, _side(kind="C"), _side(kind="P"), EXPIRIES)
    amc = forward_loop.fetch_structure_quote(symbol="AAA", structure="atm_straddle", earnings_date=date(2026, 11, 13),
                                             as_of_date=date(2026, 11, 10), release_timing="AMC")
    bmo = forward_loop.fetch_structure_quote(symbol="AAA", structure="atm_straddle", earnings_date=date(2026, 11, 13),
                                             as_of_date=date(2026, 11, 10), release_timing="BMO")
    legacy = forward_loop.fetch_structure_quote(symbol="AAA", structure="atm_straddle",
                                                earnings_date=date(2026, 11, 13), as_of_date=date(2026, 11, 10))
    assert amc["context"]["front_expiry"] == "2026-11-20"
    assert bmo["context"]["front_expiry"] == "2026-11-13"
    assert legacy["context"]["front_expiry"] == "2026-11-20"  # no timing given: treated as after the close


def test_the_wrapper_passes_release_timing_only_to_fetchers_that_take_it() -> None:
    seen = {}

    def with_timing(*, symbol, structure, earnings_date, as_of_date, context=None, release_timing=None):  # noqa: ARG001
        seen["timing"] = release_timing
        return {"mid": 1.0}

    def without(*, symbol, structure, earnings_date, as_of_date, context=None):  # noqa: ARG001
        return {"mid": 1.0}

    kw = dict(symbol="A", structure="atm_straddle", earnings_date=date(2026, 11, 13), as_of_date=date(2026, 11, 10),
              context=None, mda_client=None, release_timing="BMO")
    forward_loop._fetch_quote_for_forward_loop(with_timing, **kw)
    assert seen["timing"] == "BMO"
    assert forward_loop._fetch_quote_for_forward_loop(without, **kw)["mid"] == 1.0


@pytest.mark.parametrize("context,match", [
    ({"front_expiry": "2026-11-20", "call_strike": 102.5, "put_strike": 97.5}, True),
    ({"front_expiry": "2026-11-20", "call_strike": 105.0, "put_strike": 97.5}, False),   # strike moved
    ({"front_expiry": "2026-11-13", "call_strike": 102.5, "put_strike": 97.5}, False),   # other expiry
    ({"front_expiry": "2026-11-20", "call_strike": 102.5}, False),                      # leg missing
])
def test_compare_booked_to_scored(context, match) -> None:
    scored = {"trade_expiry": "2026-11-20",
              "structures": {"otm_strangle": {"legs": {"call": {"strike": 102.5}, "put": {"strike": 97.5}}}}}
    assert sl.compare_booked_to_scored("otm_strangle", context, scored)["match"] is match


def test_no_scored_legs_is_unknown_not_a_mismatch() -> None:
    assert sl.compare_booked_to_scored("otm_strangle", {"front_expiry": "x"}, {})["match"] is None


# ── the scorer prices the booked legs ───────────────────────────────────────


def _with_legs(structures: dict, **snapshot_kw):
    return replace(_base_snapshot(**snapshot_kw), structure_legs={"trade_expiry": "2026-04-30", "structures": structures})


def _ok(spread):
    return {"legs": {"x": {"strike": 1.0}}, "reason": None, "leg_spread_pct": spread}


@pytest.mark.parametrize("scorer,structure", [
    (score_otm_strangle, "otm_strangle"),
    (score_iron_condor, "iron_condor"),
    (score_atm_straddle, "atm_straddle"),
    (score_call_calendar, "call_calendar"),
])
def test_wider_booked_legs_cost_more_than_the_atm_proxy(scorer, structure) -> None:
    proxy = scorer(_base_snapshot())
    tight = scorer(_with_legs({structure: _ok(1.0)}))
    wide = scorer(_with_legs({structure: _ok(9.0)}))
    assert wide.execution_penalty > tight.execution_penalty
    assert wide.expected_edge_pct < tight.expected_edge_pct
    assert LEGS_UNAVAILABLE_FLAG not in wide.eligibility_flags
    assert proxy.eligible  # a snapshot without legs keeps the ATM proxy


@pytest.mark.parametrize("scorer,structure", [
    (score_otm_strangle, "otm_strangle"),
    (score_iron_condor, "iron_condor"),
    (score_atm_straddle, "atm_straddle"),
    (score_call_calendar, "call_calendar"),
])
def test_a_structure_the_chain_cannot_form_is_ineligible(scorer, structure) -> None:
    card = scorer(_with_legs({structure: {"legs": {}, "reason": "wing_strike_unavailable", "leg_spread_pct": None}}))
    assert not card.eligible and LEGS_UNAVAILABLE_FLAG in card.eligibility_flags


def test_the_snapshot_records_the_legs_on_the_trade_expiry() -> None:
    from tests.golden.test_frozen_product_cases import CASE, _snapshot

    snap = _snapshot(CASE["option_chain"])
    legs = snap.structure_legs
    assert date.fromisoformat(legs["trade_expiry"]) > snap.earnings_date  # after-close: never the same day
    assert set(legs["structures"]) == set(sl.STRUCTURES)
    assert all(entry["reason"] is None for entry in legs["structures"].values())
    assert snap.to_dict()["structure_legs"] == legs


def test_entry_records_whether_it_booked_the_scored_legs(tmp_path) -> None:
    import json

    from services import evidence_funnel as ef
    from services.baseline_evidence_store import BaselineEvidenceStore
    from services.outcome_recorder import OutcomeStore
    from services.recommendation_ledger import RecommendationLedger
    from tests.unit.quote_fixtures import IN_SESSION_QUOTES
    from tests.unit.test_services.test_quote_provenance import TODAY, _candidate

    def run(booked_strike):
        candidate = _candidate(IN_SESSION_QUOTES)
        expiry = candidate.vol_snapshot["earnings_date"]
        candidate.vol_snapshot["structure_legs"] = {
            "trade_expiry": expiry,
            "structures": {"atm_straddle": {"legs": {"call": {"strike": 100.0}, "put": {"strike": 100.0}},
                                            "reason": None, "leg_spread_pct": 1.0}},
        }
        store = OutcomeStore(store_path=tmp_path / f"o{booked_strike}.sqlite")

        def price_fetcher(*, symbol, structure, earnings_date, as_of_date, context=None):  # noqa: ARG001
            return {"mid": 5.0, "context": {"front_expiry": expiry, "strike": booked_strike,
                                             "straddle_put_strike": booked_strike}}

        funnel = ef.EvidenceFunnelStore(tmp_path / f"f{booked_strike}.sqlite")
        forward_loop.run_forward_screener(
            today=TODAY, store=store, ledger=RecommendationLedger(ledger_path=tmp_path / f"r{booked_strike}.sqlite"),
            baseline_store=BaselineEvidenceStore(tmp_path / f"b{booked_strike}.sqlite"),
            log_path=tmp_path / "log.jsonl",
            screener_builder=lambda **_: {"rows": [{"symbol": "GOOD", "status": "ranked", "dte": 6}]},
            analyzer=lambda symbol, mda_client=None: candidate, price_fetcher=price_fetcher,
            funnel=funnel, cycle_run_id="c", symbols=["GOOD"],
        )
        funnel.close()
        rows = store._conn.execute("SELECT notes FROM outcome_trades").fetchall()
        assert len(rows) == 1
        return json.loads(rows[0][0])["scored_legs_check"]

    assert run(100.0)["match"] is True
    moved = run(105.0)
    assert moved["match"] is False and moved["booked_strikes"]["call"] == 105.0
