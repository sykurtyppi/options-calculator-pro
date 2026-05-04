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

