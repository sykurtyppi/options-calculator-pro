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
    calendar = sl.discover_legs("call_calendar", calls, puts, 100.4, back_calls=_pricier(calls),
                                back_puts=_pricier(puts))
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


def _pricier(frame):
    out = frame.copy()
    out[["bid", "ask"]] = out[["bid", "ask"]] * 1.5
    return out


def _install(monkeypatch, calls, puts, expiries):
    class _Ticker:
        options = expiries

        def option_chain(self, expiry):
            if expiry == expiries[0]:
                return SimpleNamespace(calls=calls, puts=puts)
            return SimpleNamespace(calls=_pricier(calls), puts=_pricier(puts))

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
        "structures": {structure: sl.discover_legs(structure, calls, puts, 100.0, back_calls=calls, back_puts=puts,
                                                   ) if not structure.endswith("calendar") else
                       sl.discover_legs(structure, calls, puts, 100.0, back_calls=_pricier(calls),
                                        back_puts=_pricier(puts))},
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


def _with_legs(structures: dict, coverage: str = "complete", **snapshot_kw):
    return replace(_base_snapshot(**snapshot_kw), structure_legs={
        "trade_expiry": "2026-04-30", "structures": structures, "coverage": coverage})


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
    for name in ("atm_straddle", "otm_strangle", "iron_condor"):
        assert legs["structures"][name]["reason"] is None, name
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



@pytest.mark.parametrize("scorer,structure", [
    (score_otm_strangle, "otm_strangle"),
    (score_iron_condor, "iron_condor"),
    (score_atm_straddle, "atm_straddle"),
    (score_call_calendar, "call_calendar"),
])
def test_sampled_legs_neither_price_nor_rule_out_a_structure(scorer, structure) -> None:
    # Legs from the analysis chain (narrow strike window, sampled tenors) may
    # miss what entry would find: keep the ATM proxy and rule nothing out.
    proxy = scorer(_base_snapshot())
    unformable = scorer(_with_legs({structure: {"legs": {}, "reason": "wing_strike_unavailable",
                                                "leg_spread_pct": None}}, coverage="sampled"))
    wide = scorer(_with_legs({structure: _ok(9.0)}, coverage="sampled"))
    assert unformable.eligible and LEGS_UNAVAILABLE_FLAG not in unformable.eligibility_flags
    assert unformable.execution_penalty == wide.execution_penalty == proxy.execution_penalty


# ── coverage through the actual adapter and collectors (Hermes's cases) ─────


from services.earnings_vol_snapshot import _structure_legs_from_inputs, _normalize_option_chain  # noqa: E402
from services.yfinance_market_data_client import YFinanceMarketDataClient  # noqa: E402
from web.api import edge_engine  # noqa: E402

DENSE = np.arange(440.0, 560.5, 0.5)  # a 0.50 grid around spot 500


def _dense_side(kind: str, spot: float = 500.0, expiry: str = "2026-11-20") -> pd.DataFrame:
    # Later expiries cost more (time value), as on a real chain.
    tenor = 1.0 + (date.fromisoformat(expiry) - date(2026, 11, 1)).days / 30.0
    mid = np.maximum(0.5, 20.0 - np.abs(DENSE - spot) * 0.3) * tenor
    return pd.DataFrame({"contractSymbol": [f"X{kind}{k}" for k in DENSE], "strike": DENSE,
                         "bid": mid - 0.05, "ask": mid + 0.05, "lastPrice": mid, "volume": 10,
                         "openInterest": 100, "impliedVolatility": 0.40})


def _dense_ticker(expiries, spot=500.0):
    class _T:
        options = list(expiries)
        fast_info = {"last_price": spot}

        def option_chain(self, expiry):
            return SimpleNamespace(calls=_dense_side("C", spot, expiry), puts=_dense_side("P", spot, expiry))

    return _T()


WEEKLY = ["2026-11-20", "2026-11-27", "2026-12-04", "2026-12-11", "2026-12-18", "2026-12-24", "2026-12-31",
          "2027-01-08", "2027-01-15", "2027-01-22", "2027-02-19", "2027-03-19", "2027-06-17"]


def test_a_narrow_analysis_window_is_not_read_as_an_unformable_strangle() -> None:
    ticker = _dense_ticker(WEEKLY)
    client = YFinanceMarketDataClient(ticker_factory=lambda _s: ticker)
    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)

    narrow = client.get_option_chain("AAA", expiration="all", strike_limit=10)  # what the analysis fetches
    sampled, _ = _normalize_option_chain(narrow)
    only_sampled = _structure_legs_from_inputs(None, sampled, 500.0, earnings, "AMC", as_of)
    assert only_sampled["coverage"] == "sampled"
    assert only_sampled["structures"]["otm_strangle"]["reason"] == "wing_strike_unavailable"
    assert only_sampled["structures"]["iron_condor"]["reason"] == "wing_strike_unavailable"

    structure_chain = edge_engine._fetch_structure_chain(
        "AAA", client=client, use_provider=True, ticker=ticker, earnings_date=earnings,
        release_timing="AMC", as_of_date=as_of)
    complete = _structure_legs_from_inputs(structure_chain, sampled, 500.0, earnings, "AMC", as_of)
    assert complete["coverage"] == "complete"
    for structure in ("otm_strangle", "iron_condor", "atm_straddle", "call_calendar"):
        assert complete["structures"][structure]["reason"] is None, structure


@pytest.mark.parametrize("structure", ["atm_straddle", "otm_strangle", "iron_condor", "call_calendar", "put_calendar"])
def test_the_analysis_scores_what_entry_books_through_the_same_adapter(monkeypatch, structure) -> None:
    ticker = _dense_ticker(WEEKLY)
    client = YFinanceMarketDataClient(ticker_factory=lambda _s: ticker)
    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)
    monkeypatch.setattr(forward_loop, "record_provider_telemetry", lambda **_k: None)
    structure_chain = edge_engine._fetch_structure_chain(
        "AAA", client=client, use_provider=True, ticker=ticker, earnings_date=earnings,
        release_timing="AMC", as_of_date=as_of)
    scored = _structure_legs_from_inputs(structure_chain, pd.DataFrame(), 500.0, earnings, "AMC", as_of)
    quote = forward_loop.fetch_structure_quote(symbol="AAA", structure=structure, earnings_date=earnings,
                                               as_of_date=as_of, mda_client=client, release_timing="AMC")
    assert quote.get("mid") is not None, quote.get("reason")
    assert sl.compare_booked_to_scored(structure, quote["context"], scored)["match"] is True


def _dense_listing():
    # Mon/Wed/Fri expiries for three weeks, then weeklies and monthlies.
    days = [date(2026, 12, 11) + pd.Timedelta(days=i).to_pytimedelta() for i in range(22)]
    dailies = [d for d in days if d.weekday() in (0, 2, 4) and d != date(2026, 12, 25)]
    later = [date(2026, 12, 24), date(2027, 1, 8), date(2027, 1, 15), date(2027, 1, 22), date(2027, 1, 29),
             date(2027, 2, 19), date(2027, 3, 19), date(2027, 6, 17)]
    return [d.isoformat() for d in sorted(set(dailies + later))]


@pytest.mark.parametrize("earnings,front,back,sample_front,sample_back", [
    # The 6-tenor sample (chosen for IV interpolation) keeps the front but not the back expiry.
    (date(2026, 12, 16), "2026-12-18", "2027-01-01", "2026-12-18", "2027-01-15"),
    # The sample has nothing between 12-21 and 01-15: it would score a mid-January front.
    (date(2026, 12, 22), "2026-12-23", "2027-01-08", "2027-01-15", "2027-01-29"),
])
def test_trade_and_back_expiries_come_from_the_complete_listing(earnings, front, back, sample_front,
                                                                sample_back) -> None:
    ticker = _dense_ticker(_dense_listing())
    as_of = date(2026, 12, 11)
    sampled_frame = edge_engine._collect_yf_option_chain_frame(ticker, as_of_date=as_of)
    sampled, _ = _normalize_option_chain(sampled_frame)
    from_sample = _structure_legs_from_inputs(None, sampled, 500.0, earnings, "AMC", as_of)
    assert (from_sample["trade_expiry"], from_sample["back_expiry"], from_sample["coverage"]) == (
        sample_front, sample_back, "sampled")

    structure_chain = edge_engine._fetch_structure_chain(
        "AAA", client=None, use_provider=False, ticker=ticker, earnings_date=earnings,
        release_timing="AMC", as_of_date=as_of)
    complete = _structure_legs_from_inputs(structure_chain, sampled, 500.0, earnings, "AMC", as_of)
    assert (complete["trade_expiry"], complete["back_expiry"], complete["coverage"]) == (front, back, "complete")
    assert complete["structures"]["call_calendar"]["reason"] is None


def test_without_a_listing_the_legs_stay_sampled() -> None:
    class _Empty:
        options: list = []

    assert edge_engine._fetch_structure_chain("AAA", client=None, use_provider=False, ticker=_Empty(),
                                              earnings_date=date(2026, 11, 19), release_timing="AMC",
                                              as_of_date=date(2026, 11, 16)) is None
    assert edge_engine._fetch_structure_chain("AAA", client=None, use_provider=False, ticker=_dense_ticker(WEEKLY),
                                              earnings_date=None, release_timing="AMC",
                                              as_of_date=date(2026, 11, 16)) is None


def test_a_provider_chain_missing_a_side_falls_back_to_yfinance_like_entry() -> None:
    ticker = _dense_ticker(WEEKLY)

    class _CallsOnly:
        def get_expirations(self, _s):
            return WEEKLY

        def get_option_chain(self, _s, **_k):
            frame = YFinanceMarketDataClient(ticker_factory=lambda _x: ticker).get_option_chain(
                "AAA", expiration="2026-11-20", strike_limit=80)
            return frame[frame["side"] == "call"]

    structure_chain = edge_engine._fetch_structure_chain(
        "AAA", client=_CallsOnly(), use_provider=True, ticker=ticker, earnings_date=date(2026, 11, 19),
        release_timing="AMC", as_of_date=date(2026, 11, 16))
    frame, _ = _normalize_option_chain(structure_chain["chain"][0])
    assert set(frame["call_put"]) == {"C", "P"}
    assert len(frame) == 2 * len(DENSE)  # the full yfinance chain, as entry's fallback uses


def test_the_analysis_passes_the_entry_equivalent_chain_to_the_snapshot(monkeypatch) -> None:
    from unittest import mock

    import tests.unit.test_web.test_analyze_single_ticker_golden as golden

    sentinel = {"listing": ["2026-01-01"], "chain": []}
    mocks = {}
    real_patch = golden.patch

    class _Recorder:
        def __getattr__(self, name):
            return getattr(real_patch, name)

        def __call__(self, *a, **k):
            return real_patch(*a, **k)

        def object(self, target, attribute, *a, **k):
            context = real_patch.object(target, attribute, *a, **k)
            if attribute != "build_vol_snapshot":
                return context

            class _Recording:
                def __enter__(self_inner):
                    mocks["build"] = context.__enter__()
                    return mocks["build"]

                def __exit__(self_inner, *exc):
                    return context.__exit__(*exc)

            return _Recording()

    monkeypatch.setattr(golden, "patch", _Recorder())
    monkeypatch.setattr(edge_engine, "_fetch_structure_chain", lambda *a, **k: sentinel)
    golden._run_watch_scenario()
    assert isinstance(mocks["build"], mock.MagicMock)
    assert mocks["build"].call_args.kwargs["structure_chain"] is sentinel


def test_a_failed_front_fetch_never_makes_the_legs_complete() -> None:
    ticker = _dense_ticker(WEEKLY)

    class _FrontFails:
        options = WEEKLY

        def option_chain(self, expiry):
            if expiry == "2026-11-20":
                raise RuntimeError("timeout")
            return ticker.option_chain(expiry)

    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)
    assert edge_engine._fetch_structure_chain("AAA", client=None, use_provider=False, ticker=_FrontFails(),
                                              earnings_date=earnings, release_timing="AMC", as_of_date=as_of) is None
    # And a partial chain handed in directly is not complete either: only the back expiry arrived.
    back_only = edge_engine._collect_yf_option_chain_frame(ticker, as_of_date=as_of, expiries=["2026-12-04"])
    legs = _structure_legs_from_inputs({"listing": WEEKLY, "chain": [back_only]}, pd.DataFrame(), 500.0,
                                       earnings, "AMC", as_of)
    assert legs == {} or legs.get("coverage") != "complete"


def test_a_missing_back_chain_is_not_complete_coverage() -> None:
    ticker = _dense_ticker(WEEKLY)
    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)
    front_only = edge_engine._collect_yf_option_chain_frame(ticker, as_of_date=as_of, expiries=["2026-11-20"])
    legs = _structure_legs_from_inputs({"listing": WEEKLY, "chain": [front_only]}, pd.DataFrame(), 500.0,
                                       earnings, "AMC", as_of)
    assert legs == {} or legs.get("coverage") != "complete"


def test_discovery_applies_entrys_condor_opening_gates() -> None:
    calls, puts = _side(kind="C"), _side(kind="P")
    # Make the protective wings as expensive as the shorts: no credit.
    for frame, wing in ((calls, 105.0), (puts, 95.0)):
        short = 102.5 if frame is calls else 97.5
        frame.loc[frame["strike"] == wing, ["bid", "ask"]] = frame.loc[frame["strike"] == short, ["bid", "ask"]].values
    assert sl.discover_legs("iron_condor", calls, puts, 100.0)["reason"] == "non_positive_condor_credit"


def test_discovery_rejects_a_calendar_without_a_positive_debit() -> None:
    calls = _side(kind="C")
    cheap_back = calls.copy()
    cheap_back[["bid", "ask"]] = cheap_back[["bid", "ask"]] * 0.5
    result = sl.discover_legs("call_calendar", calls, _side(kind="P"), 100.0, back_calls=cheap_back, back_puts=cheap_back)
    assert result["reason"] == "non_positive_calendar_debit"


# ── review round 2: source state machine, containment, as-of guard ─────────


def _provider_frame(expiry: str, mid_scale: float, sides=("call", "put")) -> pd.DataFrame:
    frames = []
    for side in sides:
        part = _dense_side("C" if side == "call" else "P", 500.0, expiry)
        mid = np.maximum(0.5, 20.0 - np.abs(DENSE - 500.0) * 0.3) * mid_scale
        frames.append(pd.DataFrame({"optionSymbol": part["contractSymbol"], "side": side, "strike": DENSE,
                                    "bid": mid - 0.05, "ask": mid + 0.05, "mid": mid,
                                    "impliedVolatility": 0.4, "expiration_date": expiry, "underlyingPrice": 500.0}))
    return pd.concat(frames, ignore_index=True)


class _MixedProvider:
    """Front chain lacks puts (so both paths fall back to Yahoo); back chain is cheap."""

    provider_name = "fake_provider"

    def __init__(self, front, back):
        self.front, self.back = front, back

    def is_available(self):  # so entry's client resolver really uses this fake
        return True

    def get_expirations(self, _s):
        return WEEKLY

    def get_quote(self, _s):
        return 500.0

    def get_option_chain(self, _s, expiration=None, **_k):
        if expiration == self.front:
            return _provider_frame(expiration, 1.0, sides=("call",))
        return _provider_frame(expiration, 0.1)  # far cheaper than Yahoo's back


@pytest.mark.parametrize("structure", ["call_calendar", "put_calendar"])
def test_the_back_chain_comes_from_the_same_source_as_the_front_like_entry(monkeypatch, structure) -> None:
    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)
    ticker = _dense_ticker(WEEKLY)
    client = _MixedProvider("2026-11-20", "2026-12-04")
    monkeypatch.setattr(forward_loop.yf, "Ticker", lambda _s: ticker)
    monkeypatch.setattr(forward_loop, "record_provider_telemetry", lambda **_k: None)
    monkeypatch.setattr(forward_loop, "_latest_spot_price", lambda _s: 500.0)
    structure_chain = edge_engine._fetch_structure_chain(
        "AAA", client=client, use_provider=True, ticker=ticker, earnings_date=earnings,
        release_timing="AMC", as_of_date=as_of)
    scored = _structure_legs_from_inputs(structure_chain, pd.DataFrame(), 500.0, earnings, "AMC", as_of)
    assert scored["coverage"] == "complete"
    assert scored["structures"][structure]["reason"] is None  # Yahoo front + Yahoo back: a real debit
    quote = forward_loop.fetch_structure_quote(symbol="AAA", structure=structure, earnings_date=earnings,
                                               as_of_date=as_of, mda_client=client, release_timing="AMC")
    assert quote.get("mid") is not None and quote["mid"] > 0
    assert quote["quote_source"] == "yfinance"  # the provider front was one-sided: both fell back
    assert sl.compare_booked_to_scored(structure, quote["context"], scored)["match"] is True


def test_a_provider_front_keeps_the_back_on_the_provider() -> None:
    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)

    class _Provider(_MixedProvider):
        def get_option_chain(self, _s, expiration=None, **_k):
            return _provider_frame(expiration, 1.0 if expiration == self.front else 2.0)

    calls = []

    class _NoYahoo:
        def option_chain(self, expiry):  # must not be used: the front came from the provider
            calls.append(expiry)
            raise AssertionError("yahoo used")

    chain = edge_engine._fetch_structure_chain("AAA", client=_Provider("2026-11-20", "2026-12-04"),
                                               use_provider=True, ticker=_NoYahoo(), earnings_date=earnings,
                                               release_timing="AMC", as_of_date=as_of)
    assert chain is not None and calls == []


def test_an_unreadable_yahoo_listing_does_not_break_an_explicit_expiry_fetch() -> None:
    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)
    base = _dense_ticker(WEEKLY)

    class _ListingRaises:
        @property
        def options(self):
            raise RuntimeError("Yahoo listing unavailable")

        def option_chain(self, expiry):
            return base.option_chain(expiry)

    chain = edge_engine._fetch_structure_chain(
        "AAA", client=_MixedProvider("2026-11-20", "2026-12-04"), use_provider=True, ticker=_ListingRaises(),
        earnings_date=earnings, release_timing="AMC", as_of_date=as_of)
    assert chain is not None
    legs = _structure_legs_from_inputs(chain, pd.DataFrame(), 500.0, earnings, "AMC", as_of)
    assert legs["coverage"] == "complete"


def test_a_failing_structure_fetch_never_aborts_the_analysis(monkeypatch) -> None:
    from tests.unit.test_web.test_analyze_single_ticker_golden import _run_watch_scenario

    def boom(*_a, **_k):
        raise RuntimeError("listing unavailable")

    monkeypatch.setattr(edge_engine, "_fetch_structure_chain", boom)
    result = _run_watch_scenario()
    assert result.recommendation  # the analysis completed


def test_structure_quotes_obey_the_as_of_boundary() -> None:
    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)
    ticker = _dense_ticker(WEEKLY)
    fresh = edge_engine._fetch_structure_chain("AAA", client=None, use_provider=False, ticker=ticker,
                                               earnings_date=earnings, release_timing="AMC",
                                               as_of_date=as_of)
    future = {"listing": fresh["listing"],
              "chain": [part.assign(trade_date="2026-11-18") for part in fresh["chain"]]}
    undated = {"listing": fresh["listing"], "chain": [part.assign(trade_date=None) for part in fresh["chain"]]}
    for chain in (future, undated):
        legs = _structure_legs_from_inputs(chain, pd.DataFrame(), 500.0, earnings, "AMC", as_of)
        assert legs == {} or legs.get("coverage") != "complete"
    on_time = {"listing": fresh["listing"], "chain": [part.assign(trade_date="2026-11-16") for part in fresh["chain"]]}
    assert _structure_legs_from_inputs(on_time, pd.DataFrame(), 500.0, earnings, "AMC", as_of)["coverage"] == "complete"


@pytest.mark.parametrize("with_empty_column", [False, True])
def test_undated_live_rows_are_stamped_with_the_fetch_date(monkeypatch, with_empty_column) -> None:
    monkeypatch.setattr(edge_engine, "_market_today_date", lambda: date(2026, 11, 16))
    earnings = date(2026, 11, 19)

    class _Provider(_MixedProvider):
        def get_option_chain(self, _s, expiration=None, **_k):
            frame = _provider_frame(expiration, 1.0 if expiration == self.front else 2.0)
            return frame.assign(trade_date=None) if with_empty_column else frame

    # Analysed as of an EARLIER day than the fetch (e.g. a re-run): rows carry the
    # fetch date, never the as-of date, so the as-of guard refuses them.
    chain = edge_engine._fetch_structure_chain(
        "AAA", client=_Provider("2026-11-20", "2026-12-04"), use_provider=True, ticker=None,
        earnings_date=earnings, release_timing="AMC", as_of_date=date(2026, 11, 13))
    assert chain["fetched_on"] == "2026-11-16"
    past = _structure_legs_from_inputs(chain, pd.DataFrame(), 500.0, earnings, "AMC", date(2026, 11, 13))
    assert past == {} or past.get("coverage") != "complete"
    today = _structure_legs_from_inputs(chain, pd.DataFrame(), 500.0, earnings, "AMC", date(2026, 11, 16))
    assert today["coverage"] == "complete"


@pytest.mark.parametrize("column,value", [
    ("report_date", "2026-11-18"),
    ("updated", pd.Timestamp("2026-11-18 15:00", tz="UTC").timestamp()),
])
def test_a_providers_own_quote_time_is_never_overwritten_by_the_fetch_date(monkeypatch, column, value) -> None:
    monkeypatch.setattr(edge_engine, "_market_today_date", lambda: date(2026, 11, 16))
    earnings = date(2026, 11, 19)

    class _Dated(_MixedProvider):
        def get_option_chain(self, _s, expiration=None, **_k):
            return _provider_frame(expiration, 1.0 if expiration == self.front else 2.0).assign(**{column: value})

    chain = edge_engine._fetch_structure_chain(
        "AAA", client=_Dated("2026-11-20", "2026-12-04"), use_provider=True, ticker=None,
        earnings_date=earnings, release_timing="AMC", as_of_date=date(2026, 11, 16))
    # Quotes dated 11-18 analysed as of 11-16: refused, whatever the fetch clock says.
    legs = _structure_legs_from_inputs(chain, pd.DataFrame(), 500.0, earnings, "AMC", date(2026, 11, 16))
    assert legs == {} or legs.get("coverage") != "complete"
    later = _structure_legs_from_inputs(chain, pd.DataFrame(), 500.0, earnings, "AMC", date(2026, 11, 18))
    assert later["coverage"] == "complete"


def test_the_yahoo_structure_fetch_does_not_stamp_the_as_of_date() -> None:
    frame = edge_engine._collect_yf_option_chain_frame(_dense_ticker(WEEKLY), as_of_date=date(2026, 11, 13),
                                                       expiries=["2026-11-20"], stamp_trade_date=False)
    assert frame["trade_date"].isna().all()
    stamped = edge_engine._collect_yf_option_chain_frame(_dense_ticker(WEEKLY), as_of_date=date(2026, 11, 13),
                                                         expiries=["2026-11-20"])
    assert set(stamped["trade_date"]) == {"2026-11-13"}  # the term-structure path is unchanged


@pytest.mark.parametrize("failing", ["2026-11-20", "2026-12-04"])
def test_a_provider_error_leaves_the_legs_sampled_as_entry_fails(monkeypatch, failing) -> None:
    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)

    class _Raises(_MixedProvider):
        def get_option_chain(self, _s, expiration=None, **_k):
            if expiration == failing:
                raise TimeoutError("provider timeout")
            return _provider_frame(expiration, 1.0 if expiration == self.front else 2.0)

    client = _Raises("2026-11-20", "2026-12-04")
    ticker = _dense_ticker(WEEKLY)
    assert edge_engine._fetch_structure_chain("AAA", client=client, use_provider=True, ticker=ticker,
                                              earnings_date=earnings, release_timing="AMC",
                                              as_of_date=as_of) is None
    monkeypatch.setattr(forward_loop.yf, "Ticker", lambda _s: ticker)
    monkeypatch.setattr(forward_loop, "record_provider_telemetry", lambda **_k: None)
    structure = "call_calendar"
    with pytest.raises(TimeoutError):  # entry does not survive it either
        forward_loop.fetch_structure_quote(symbol="AAA", structure=structure, earnings_date=earnings,
                                           as_of_date=as_of, mda_client=client, release_timing="AMC")


def test_a_provider_front_and_back_are_booked_from_the_provider(monkeypatch) -> None:
    earnings, as_of = date(2026, 11, 19), date(2026, 11, 16)

    class _Provider(_MixedProvider):
        def get_option_chain(self, _s, expiration=None, **_k):
            return _provider_frame(expiration, 1.0 if expiration == self.front else 2.0)

    client = _Provider("2026-11-20", "2026-12-04")
    monkeypatch.setattr(forward_loop, "record_provider_telemetry", lambda **_k: None)
    chain = edge_engine._fetch_structure_chain("AAA", client=client, use_provider=True, ticker=None,
                                               earnings_date=earnings, release_timing="AMC", as_of_date=as_of)
    scored = _structure_legs_from_inputs(chain, pd.DataFrame(), 500.0, earnings, "AMC", as_of)
    quote = forward_loop.fetch_structure_quote(symbol="AAA", structure="call_calendar", earnings_date=earnings,
                                               as_of_date=as_of, mda_client=client, release_timing="AMC")
    assert quote["quote_source"] == "fake_provider"
    assert sl.compare_booked_to_scored("call_calendar", quote["context"], scored)["match"] is True


def test_discovery_rejects_a_condor_credit_at_or_above_its_wing_width() -> None:
    calls, puts = _side(kind="C"), _side(kind="P")
    # Make the shorts worth far more than the wings: credit >= width (broken quotes).
    for frame, short in ((calls, 102.5), (puts, 97.5)):
        frame.loc[frame["strike"] == short, ["bid", "ask"]] = [9.9, 10.1]
    assert sl.discover_legs("iron_condor", calls, puts, 100.0)["reason"] == "condor_credit_exceeds_wing_width"


def test_a_failed_back_fetch_leaves_the_legs_sampled() -> None:
    base = _dense_ticker(WEEKLY)

    class _BackFails:
        options = WEEKLY

        def option_chain(self, expiry):
            if expiry == "2026-12-04":
                raise RuntimeError("timeout")
            return base.option_chain(expiry)

    assert edge_engine._fetch_structure_chain("AAA", client=None, use_provider=False, ticker=_BackFails(),
                                              earnings_date=date(2026, 11, 19), release_timing="AMC",
                                              as_of_date=date(2026, 11, 16)) is None


def test_a_later_live_yahoo_fetch_is_refused_for_an_earlier_as_of(monkeypatch) -> None:
    monkeypatch.setattr(edge_engine, "_market_today_date", lambda: date(2026, 11, 16))
    earnings = date(2026, 11, 19)
    chain = edge_engine._fetch_structure_chain("AAA", client=None, use_provider=False, ticker=_dense_ticker(WEEKLY),
                                               earnings_date=earnings, release_timing="AMC",
                                               as_of_date=date(2026, 11, 13))
    past = _structure_legs_from_inputs(chain, pd.DataFrame(), 500.0, earnings, "AMC", date(2026, 11, 13))
    assert past == {} or past.get("coverage") != "complete"
    assert _structure_legs_from_inputs(chain, pd.DataFrame(), 500.0, earnings, "AMC",
                                       date(2026, 11, 16))["coverage"] == "complete"
