"""Alert evidence (review of the 2026-10-12 TSLA alert).

The score fills missing inputs with neutral defaults, so a finite score could
hide missing term structure, move history or spread; the ATM IV check was a
1% floor only; an unconfirmed or stale earnings date passed silently; the
message implied a "premarket" setup-quality call; and nothing but the symbol
was kept, so a sent alert could not be reconstructed.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest

from scripts.premarket_screener_alert import (
    MAX_ATM_SPREAD_PCT,
    format_alert_message,
    main,
    score_breakdown,
    select_qualifying_rows,
)
from services.screener_service import compute_ranking_score, ranking_score_components


def _row(symbol, score=0.80, **extra):
    row = {
        "symbol": symbol, "ranking_score": score, "days_to_earnings": 9, "release_timing": "AMC",
        "iv_rv_ratio": 1.15, "iv30": 0.55, "avg_spread_pct": 2.0, "term_structure_ratio": 1.0,
        "median_earnings_move_pct": 7.0, "sample_size": 12, "earnings_source_confirmed": True,
        "earnings_source_stale": False,
    }
    row.update(extra)
    return row


def _select(rows, dropped=None):
    return select_qualifying_rows(rows, min_score=0.65, top_n=9, dte_min=3, dte_max=10, dropped=dropped)


# ── hard evidence gates ──────────────────────────────────────────────────────


@pytest.mark.parametrize("extra, reason", [
    ({"avg_spread_pct": MAX_ATM_SPREAD_PCT + 0.1}, "spread_too_wide"),
    ({"avg_spread_pct": None}, "missing_spread"),
    ({"avg_spread_pct": -1.0}, "missing_spread"),
    ({"term_structure_ratio": None}, "missing_term_structure"),
    ({"median_earnings_move_pct": float("nan")}, "missing_move_history"),
    ({"iv30": 5.5}, "implausible_or_missing_atm_iv"),
    ({"earnings_source_stale": True}, "stale_earnings_date"),
])
def test_each_missing_or_implausible_input_is_dropped_with_its_reason(extra, reason):
    dropped = {}
    assert _select([_row("BAD", **extra)], dropped) == []
    assert dropped == {reason: 1}


def test_boundary_values_still_qualify():
    rows = [_row("EDGE", avg_spread_pct=MAX_ATM_SPREAD_PCT, iv30=5.0)]
    assert [r["symbol"] for r in _select(rows)] == ["EDGE"]


# ── the message ──────────────────────────────────────────────────────────────


def test_an_unconfirmed_earnings_date_is_labelled_not_dropped():
    rows = [_row("CONF"), _row("UNCONF", earnings_source_confirmed=None),
            _row("FALSE", earnings_source_confirmed=False)]
    assert [r["symbol"] for r in _select(rows)] == ["CONF", "UNCONF", "FALSE"]
    lines = format_alert_message(rows, as_of=date(2026, 10, 12), min_score=0.65).splitlines()
    by_symbol = {line.split()[1]: line for line in lines if line[:1].isdigit()}
    assert "[date unconfirmed]" not in by_symbol["CONF"]
    assert "[date unconfirmed]" in by_symbol["UNCONF"] and "[date unconfirmed]" in by_symbol["FALSE"]


def test_regime_flag_names_the_iv_rv_that_was_scored():
    row = _row("PYPL", iv_rv_ratio=0.80, iv_rv_har=1.42, rv_percentile_rank=100.0, iv_regime_conditioned=True)
    msg = format_alert_message([row], as_of=date(2026, 10, 12), min_score=0.65)
    assert "IV/RV 0.80 [regime: scored IV/RV 1.42]" in msg


@pytest.mark.parametrize("built_at, stamp", [
    (datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc), "Built 11:00 EDT"),
    (datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc), "Built 10:00 EST"),
])
def test_message_says_it_is_a_candidate_built_after_the_open(built_at, stamp):
    msg = format_alert_message([_row("TSLA")], as_of=built_at.date(), min_score=0.65, built_at=built_at)
    assert "Pre-earnings long-vol candidates (heuristic score >= 0.65)" in msg
    assert "Selector approval still required" in msg and "not a win probability" in msg
    assert stamp in msg and "after the US open" in msg
    assert "not financial advice" in msg.lower() and len(msg) < 1000


# ── score components ─────────────────────────────────────────────────────────


def test_components_sum_to_the_score_and_name_neutral_fallbacks():
    full = ranking_score_components(1.15, 1.0, 7.0, 12, 9, 2.0)
    assert full["score"] == compute_ranking_score(1.15, 1.0, 7.0, 12, 9, 2.0)
    assert sum(full["components"].values()) == pytest.approx(full["score"])
    assert full["neutral_fallbacks"] == [] and full["scored_iv_rv"] == 1.15
    sparse = ranking_score_components(1.15, None, None, 12, 9, None)
    assert sparse["neutral_fallbacks"] == ["move_history", "term_structure", "liquidity"]


# ── audit trail ──────────────────────────────────────────────────────────────


def _table(rows):
    return {"rows": rows, "universe_size": 187, "in_entry_window": len(rows)}


def test_a_sent_alert_is_audited_in_full(tmp_path):
    audit = tmp_path / "audit.jsonl"
    rows = [_row("TSLA", earnings_source_confirmed=None), _row("WIDE", avg_spread_pct=40.0)]
    with patch("scripts.premarket_screener_alert.build_ranked_screener", return_value=_table(rows)), \
            patch("scripts.premarket_screener_alert.IMessageConfig.from_env", return_value=object()), \
            patch("scripts.premarket_screener_alert.send_imessage", return_value={"to": "***1"}) as send:
        assert main(["--state-path", str(tmp_path / "s.json"), "--audit-path", str(audit)]) == 0
    assert send.call_count == 1
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["dropped"] == {"spread_too_wide": 1}
    assert record["screened"]["universe_size"] == 187 and record["gates"]["max_atm_spread_pct"] == 15.0
    assert "[date unconfirmed]" in record["message"]
    (setup,) = record["setups"]
    assert setup["row"]["symbol"] == "TSLA"
    breakdown = setup["score_breakdown"]
    assert set(breakdown["components"]) == {"iv_entry", "move_history", "term_structure", "dte", "sample",
                                            "liquidity"}
    assert breakdown == json.loads(json.dumps(score_breakdown(rows[0])))


def test_a_dry_run_writes_no_audit(tmp_path):
    audit = tmp_path / "audit.jsonl"
    with patch("scripts.premarket_screener_alert.build_ranked_screener", return_value=_table([_row("TSLA")])), \
            patch("scripts.premarket_screener_alert.send_imessage") as send:
        assert main(["--dry-run", "--state-path", str(tmp_path / "s.json"), "--audit-path", str(audit)]) == 0
    send.assert_not_called()
    assert not audit.exists()
