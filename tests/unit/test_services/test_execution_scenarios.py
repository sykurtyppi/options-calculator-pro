from __future__ import annotations

from services.execution_scenarios import build_execution_scenarios, compare_execution_scenarios


def test_long_straddle_entry_tracks_mid_and_spread_crossing() -> None:
    quote = {
        "bid_ask_mid": {
            "legs": {
                "call": {"bid": 1.0, "ask": 1.4, "mid": 1.2},
                "put": {"bid": 0.8, "ask": 1.0, "mid": 0.9},
            }
        }
    }

    scenarios = build_execution_scenarios(
        structure="atm_straddle",
        quote_payload=quote,
        phase="entry",
    ).to_dict()

    assert scenarios["scenario_values"]["mid"] == 2.1
    assert scenarios["scenario_values"]["cross_50"] == 2.4
    assert scenarios["spread_cost_vs_mid"]["cross_50"] == 0.3
    assert scenarios["spread_as_pct_of_premium"] == 28.571429


def test_exit_scenarios_move_long_legs_toward_bid() -> None:
    quote = {
        "bid_ask_mid": {
            "legs": {
                "call": {"bid": 1.0, "ask": 1.4, "mid": 1.2},
                "put": {"bid": 0.8, "ask": 1.0, "mid": 0.9},
            }
        }
    }

    scenarios = build_execution_scenarios(
        structure="atm_straddle",
        quote_payload=quote,
        phase="exit",
    ).to_dict()

    assert scenarios["scenario_values"]["mid"] == 2.1
    assert scenarios["scenario_values"]["cross_50"] == 1.8
    assert scenarios["spread_cost_vs_mid"]["cross_50"] == 0.3


def test_calendar_scenarios_treat_front_leg_as_short_back_leg_as_long() -> None:
    quote = {
        "bid_ask_mid": {
            "legs": {
                "front": {"bid": 1.0, "ask": 1.2, "mid": 1.1},
                "back": {"bid": 2.0, "ask": 2.4, "mid": 2.2},
            }
        }
    }

    scenarios = build_execution_scenarios(
        structure="call_calendar",
        quote_payload=quote,
        phase="entry",
    ).to_dict()

    assert scenarios["scenario_values"]["mid"] == 1.1
    assert scenarios["scenario_values"]["cross_50"] == 1.4


def test_compare_execution_scenarios_returns_per_fill_case() -> None:
    entry = {"scenario_values": {"mid": 2.0, "cross_25": 2.1, "cross_50": 2.2, "conservative": 2.2}}
    exit = {"scenario_values": {"mid": 2.4, "cross_25": 2.3, "cross_50": 2.2, "conservative": 2.2}}

    result = compare_execution_scenarios(entry=entry, exit=exit)

    assert result["realized_return_pct"]["mid"] == 20.0
    assert result["realized_return_pct"]["cross_50"] == 0.0
    assert result["realized_pnl"]["mid"] == 40.0



# ── Iron condor: credit structure, short legs fill toward the bid ────────────


def _condor_quote():
    # net credit at mid = (1.1 + 0.9) - (0.5 + 0.4) = 1.1
    return {
        "bid_ask_mid": {
            "legs": {
                "short_call": {"bid": 1.0, "ask": 1.2, "mid": 1.1},
                "long_call": {"bid": 0.4, "ask": 0.6, "mid": 0.5},
                "short_put": {"bid": 0.8, "ask": 1.0, "mid": 0.9},
                "long_put": {"bid": 0.3, "ask": 0.5, "mid": 0.4},
            }
        }
    }


def test_condor_scenario_value_is_reported_as_a_positive_credit() -> None:
    scenarios = build_execution_scenarios(
        structure="iron_condor", quote_payload=_condor_quote(), phase="entry",
    ).to_dict()
    # Reported with the same sign convention as the structure's quoted `mid`.
    assert scenarios["scenario_values"]["mid"] == 1.1


def test_condor_adverse_entry_fill_collects_less_credit() -> None:
    scenarios = build_execution_scenarios(
        structure="iron_condor", quote_payload=_condor_quote(), phase="entry",
    ).to_dict()
    # Crossing the spread on entry must REDUCE the credit received (the short
    # legs fill toward the bid, the long legs toward the ask) and register as a
    # positive execution cost.
    assert scenarios["scenario_values"]["cross_50"] < scenarios["scenario_values"]["mid"]
    assert scenarios["spread_cost_vs_mid"]["cross_50"] > 0


def test_condor_scenario_returns_are_credit_aware() -> None:
    entry = build_execution_scenarios(
        structure="iron_condor", quote_payload=_condor_quote(), phase="entry",
    ).to_dict()
    # Closing cheaper than the credit collected is a WIN for the seller.
    cheaper = {"scenario_values": {k: 0.4 for k in entry["scenario_values"]}}
    outcomes = compare_execution_scenarios(
        entry=entry, exit=cheaper, structure="iron_condor", capital_at_risk=0.9,
    )
    assert outcomes["realized_return_pct"]["mid"] > 0
    assert outcomes["realized_pnl"]["mid"] == 70.0

    # And closing more expensive is a loss.
    pricier = {"scenario_values": {k: 1.8 for k in entry["scenario_values"]}}
    losing = compare_execution_scenarios(
        entry=entry, exit=pricier, structure="iron_condor", capital_at_risk=0.9,
    )
    assert losing["realized_return_pct"]["mid"] < 0


def test_compare_execution_scenarios_default_behaviour_unchanged() -> None:
    # Without `structure` the debit math must be byte-identical to before.
    entry = {"scenario_values": {"mid": 2.0}}
    exit_ = {"scenario_values": {"mid": 2.5}}
    outcomes = compare_execution_scenarios(entry=entry, exit=exit_)
    assert outcomes["realized_return_pct"]["mid"] == 25.0
    assert outcomes["realized_pnl"]["mid"] == 50.0


def test_thin_credit_condor_keeps_its_adverse_scenarios() -> None:
    """AUDIT MEDIUM: the `entry_val <= 0` guard exists to avoid dividing by a
    non-positive premium base. A credit structure with an explicit capital_at_risk
    base never divides by entry_val, so the guard was nulling exactly the WORST
    execution scenarios for thin-credit condors — biasing the diagnostics
    optimistic precisely where execution matters most."""
    entry = {"scenario_values": {"mid": 0.40, "cross_25": -0.10, "cross_50": -0.60, "conservative": -0.60}}
    exit_ = {"scenario_values": {"mid": 0.20, "cross_25": 0.20, "cross_50": 0.20, "conservative": 0.20}}
    out = compare_execution_scenarios(
        entry=entry, exit=exit_, structure="iron_condor", capital_at_risk=1.60,
    )
    # Every scenario keeps a P&L — none are silently dropped any more.
    for name in ("mid", "cross_25", "cross_50", "conservative"):
        assert out["realized_pnl"][name] is not None, f"{name} P&L was dropped"
    # Adverse fills collected less credit, so they must show a WORSE P&L.
    assert out["realized_pnl"]["cross_50"] < out["realized_pnl"]["mid"]
    # The RETURN is withheld where the scenario implies the structure was opened
    # for a DEBIT: capital at risk is no longer the nominal max loss there, so
    # dividing by it would inflate the ratio by an order of magnitude.
    assert out["realized_return_pct"]["mid"] is not None
    assert out["realized_return_pct"]["cross_50"] is None


def test_non_positive_entry_still_guarded_without_a_risk_base() -> None:
    # Debit structures (and credit ones with no capital_at_risk) still divide by
    # entry_val, so the original guard must remain for them.
    entry = {"scenario_values": {"mid": -0.5}}
    exit_ = {"scenario_values": {"mid": 0.2}}
    assert compare_execution_scenarios(entry=entry, exit=exit_)["realized_return_pct"]["mid"] is None
    assert compare_execution_scenarios(
        entry=entry, exit=exit_, structure="iron_condor", capital_at_risk=None,
    )["realized_return_pct"]["mid"] is None
