"""Semantic regressions for the theoretical payoff diagrams (Hermes pricing review).

1. "Post-event" P&L used exactly one day to expiry whatever the real expiry;
2. the "severe" crush scenario could keep MORE IV than "mild", and the move
   proxy was labelled a historical IV calibration;
plus fail-closed behaviour when the options expire before the event, and the
disclosed calendar assumptions.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from web.api.edge_math import (
    _derive_iv_scenarios,
    _straddle_payoff,
    _strangle_payoff,
    post_event_valuation_days,
)


def _flat_pnl_at_zero_move(payoff, key="iv_crush_25"):
    row = next(r for r in payoff["payoff_scenarios"] if r["move_pct"] == 0.0)
    return row[key]


# ── 1. post-event horizon ─────────────────────────────────────────────────────


def _bsm_straddle(spot, strike, days, iv, r=0.045):
    t = days / 365.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    n = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    call = spot * n(d1) - strike * math.exp(-r * t) * n(d2)
    return call + (call - spot + strike * math.exp(-r * t))


def test_hermes_example_error_size():
    # ATM, 14d entry at 60% IV, IV 30% after the event, no move, 9 days left.
    # Valuing with 1 day left instead of 9 understates the position by about
    # $2.50/share (~$250 per contract) at r=4.5% here; the review quoted $2.85
    # with unstated rate/day-count inputs. Either way the error is material.
    one_day = _bsm_straddle(100, 100, 1, 0.30)
    nine_days = _bsm_straddle(100, 100, 9, 0.30)
    assert nine_days - one_day == pytest.approx(2.50, abs=0.01)


def test_straddle_is_valued_with_its_real_remaining_life():
    payoff = _straddle_payoff(100.0, 0.60, 14.0, valuation_days=5)
    assert (payoff["T_remain_days"], payoff["valuation_days_after_entry"]) == (9, 5)
    assert payoff["entry_debit"] == pytest.approx(_bsm_straddle(100, 100, 14, 0.60), abs=1e-4)
    expected = _bsm_straddle(100, 100, 9, 0.60) - _bsm_straddle(100, 100, 14, 0.60)
    assert _flat_pnl_at_zero_move(payoff, "iv_flat") == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("days_left", [1, 3, 7, 14])
def test_remaining_life_equals_expiry_minus_valuation(days_left):
    expiry = 20.0
    valuation = expiry - days_left
    for payoff in (
        _straddle_payoff(100.0, 0.5, expiry, valuation_days=valuation),
        _strangle_payoff(100.0, 0.5, expiry, 6.0, valuation_days=valuation),
    ):
        assert payoff["T_remain_days"] == days_left
        assert payoff["valuation_days_after_entry"] == valuation


def test_remaining_life_changes_the_value():
    # At zero move and flat IV, a straddle with more life left is worth more.
    values = [
        _flat_pnl_at_zero_move(_straddle_payoff(100.0, 0.5, 20.0, valuation_days=20 - d), "iv_flat")
        for d in (1, 3, 7, 14)
    ]
    assert values == sorted(values) and len(set(values)) == 4


@pytest.mark.parametrize("valuation", [15, 30])
def test_options_expiring_before_the_reaction_are_not_valued(valuation):
    assert _straddle_payoff(100.0, 0.5, 14.0, valuation_days=valuation) is None
    assert _strangle_payoff(100.0, 0.5, 14.0, 6.0, valuation_days=valuation) is None


def test_options_expiring_on_the_reaction_session_trade_through_it():
    # Valued at that session's open with one regular session (6.5h) left,
    # not withheld: they expire at its close, after the reaction.
    for payoff in (
        _straddle_payoff(100.0, 0.5, 14.0, valuation_days=14),
        _strangle_payoff(100.0, 0.5, 14.0, 6.0, valuation_days=14),
    ):
        assert payoff is not None
        assert payoff["T_remain_days"] == round(6.5 / 24, 2)
        assert "expiry at that session's close" in payoff["note"]


def test_positional_arguments_are_unchanged():
    # S, iv, T_near_days, r, n_points, raw_moves_pct, implied_move_pct, q
    assert _straddle_payoff(100.0, 0.5, 14.0, 0.045, 41, None, None, 0.0) is not None
    assert _strangle_payoff(100.0, 0.5, 14.0, 6.0, 0.045, 41, None, None, 0.0) is not None
    with pytest.raises(TypeError):
        _straddle_payoff(100.0, 0.5, 14.0, 0.045, 41, None, None, 0.0, 5)  # valuation_days is keyword-only


@pytest.mark.parametrize(
    ("as_of", "days_to_earnings", "timing", "expected"),
    [
        (date(2026, 4, 20), 3, "before market open", 3),   # Thu BMO -> Thursday
        (date(2026, 4, 20), 3, "during market hours", 3),
        (date(2026, 4, 20), 3, "after market close", 4),    # Thu AMC -> Friday
        (date(2026, 4, 20), 4, "after market close", 7),    # Fri AMC -> Monday
        (date(2026, 4, 20), 4, None, 7),                    # unknown timing treated as AMC
        (date(2026, 4, 20), 0, "after market close", 1),
        (date(2026, 4, 20), 5, "before market open", 7),    # Sat BMO -> Monday
        (date(2026, 4, 20), 6, "during market hours", 7),   # Sun intraday -> Monday
        (date(2026, 4, 20), 5, "after market close", 7),    # Sat AMC -> Monday
    ],
)
def test_post_event_session(as_of, days_to_earnings, timing, expected):
    assert post_event_valuation_days(as_of, days_to_earnings, timing) == expected


@pytest.mark.parametrize("bad", [None, -1, float("nan")])
def test_post_event_session_unknown(bad):
    assert post_event_valuation_days(date(2026, 4, 20), bad, "after market close") is None


# ── 2. ordered, honestly labelled IV scenarios ───────────────────────────────


def test_hermes_example_severe_crush_is_not_milder_than_mild():
    scenarios = _derive_iv_scenarios(0.40, [2, 3, 4, 5, 8, 8, 8, 8], 10.0)
    assert scenarios["iv_crush_severe"] <= scenarios["iv_crush_mild"]
    assert scenarios["iv_crush_mild"] == pytest.approx(0.33)


@pytest.mark.parametrize(
    "moves",
    [
        [2, 3, 4, 5, 8, 8, 8, 8],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [20, 25, 30, 35],
        [0.5, 9.5, 0.5, 9.5, 0.5, 9.5],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [0.1, 0.2, 12, 15, 1, 2],
    ],
)
@pytest.mark.parametrize("implied", [2.0, 5.0, 10.0, 40.0])
def test_scenarios_are_ordered(moves, implied):
    s = _derive_iv_scenarios(0.40, moves, implied)
    assert s["iv_crush_severe"] <= s["iv_crush_mild"] <= s["iv_flat"] <= s["iv_expand"]


def test_fallback_scenarios_are_ordered():
    s = _derive_iv_scenarios(0.40, None, None)
    assert s["_source"] == "heuristic_fallback"
    assert s["iv_crush_severe"] <= s["iv_crush_mild"] <= s["iv_flat"] <= s["iv_expand"]


def test_scenario_source_says_it_is_a_stock_move_proxy():
    assert _derive_iv_scenarios(0.4, [2, 3, 4, 5, 8, 8, 8, 8], 10.0)["_source"] == "historical_move_proxy"
    assert _derive_iv_scenarios(0.4, [2, 3, 4, 5], 10.0)["_source"] == "small_sample_move_proxy"
    assert "_crush_p75_pct" in _derive_iv_scenarios(0.4, [2, 3, 4, 5], 10.0)


# ── engine wiring ────────────────────────────────────────────────────────────


def test_engine_values_the_straddle_at_the_post_event_session():
    from tests.unit.test_web.test_analyze_single_ticker_golden import _run_watch_scenario

    # as_of Mon 2024-07-01, earnings Tue 07-09 after the close -> reaction
    # Wed 07-10 = 9 days after entry; a 15-day expiry then has 6 days left.
    metrics = _run_watch_scenario(near_term_dte=15).metrics
    payoff = metrics["structure_payoff"]
    assert metrics["structure_payoff_unavailable_reason"] is None
    assert payoff["structure"] == "atm_straddle"
    assert (payoff["valuation_days_after_entry"], payoff["T_remain_days"]) == (9, 6)
    assert payoff["valuation_basis"] == "first_post_event_session"


def test_engine_values_a_straddle_expiring_on_the_reaction_session():
    from tests.unit.test_web.test_analyze_single_ticker_golden import _run_watch_scenario

    # Earnings Tue after the close -> reaction Wed (9 days); a Wed expiry
    # trades through the reaction until that close.
    metrics = _run_watch_scenario(near_term_dte=9).metrics
    payoff = metrics["structure_payoff"]
    assert metrics["structure_payoff_unavailable_reason"] is None
    assert (payoff["valuation_days_after_entry"], payoff["T_remain_days"]) == (9, round(6.5 / 24, 2))
    assert metrics["calendar_payoff"]["assumptions"]["synthetic_contracts"] is True


def test_engine_withholds_a_straddle_that_expires_before_the_reaction():
    from tests.unit.test_web.test_analyze_single_ticker_golden import _run_watch_scenario

    metrics = _run_watch_scenario(near_term_dte=8).metrics  # expires Tue, reaction prints Wed
    assert metrics["structure_payoff"] is None
    assert metrics["structure_payoff_unavailable_reason"] == "near_expiry_before_earnings_reaction"
