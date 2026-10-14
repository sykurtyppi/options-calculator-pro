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
from tests.unit.test_scripts.alert_fixtures import alert_row


def _row(symbol, score=0.80, **extra):
    extra.setdefault("days_to_earnings", 9)
    dte = extra.pop("days_to_earnings")
    return alert_row(symbol, score, dte=dte, **extra)


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
    # Outbox: the full record is written as prepared BEFORE the send, then closed.
    assert [r["status"] for r in records] == ["prepared", "accepted"]
    assert records[0]["event_id"] == records[1]["event_id"]
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


# ── review of PR #166: the cases Hermes reproduced ─────────────────────────


def test_a_named_confirming_source_is_a_confirmed_date():
    """Production stores the confirming source's NAME (Optional[str])."""
    rows = [_row("SEC", earnings_source_confirmed="sec_edgar"),
            _row("NASDAQ", earnings_source_confirmed="nasdaq_calendar"),
            _row("BOOL", earnings_source_confirmed=True),
            _row("EMPTY", earnings_source_confirmed=""),
            _row("BLANK", earnings_source_confirmed="  "),
            _row("NONE", earnings_source_confirmed=None)]
    lines = format_alert_message(rows, as_of=date(2026, 10, 12), min_score=0.65).splitlines()
    labelled = {line.split()[1] for line in lines if line[:1].isdigit() and "[date unconfirmed]" in line}
    assert labelled == {"EMPTY", "BLANK", "NONE"}


@pytest.mark.parametrize("value", [0, -1.0, 0.0, float("inf"), float("nan"), "1.0", True, None])
def test_term_structure_must_be_a_positive_number(value):
    dropped = {}
    assert _select([_row("TS", term_structure_ratio=value)], dropped) == []
    assert dropped == {"missing_term_structure": 1}


@pytest.mark.parametrize("value", [-0.5, float("nan"), "6.0", False])
def test_move_history_must_be_a_non_negative_number(value):
    dropped = {}
    assert _select([_row("MOVE", median_earnings_move_pct=value)], dropped) == []
    assert dropped == {"missing_move_history": 1}


@pytest.mark.parametrize("value", ["nan", float("nan"), -1, 7.5, "8", True])
def test_sample_size_must_be_a_whole_count(value):
    dropped = {}
    assert _select([_row("SAMPLE", sample_size=value)], dropped) == []
    assert dropped == {"invalid_sample_size": 1}


def test_a_score_its_inputs_do_not_produce_is_rejected():
    row = _row("ODD")
    row["ranking_score"] = 0.99
    dropped = {}
    assert _select([row], dropped) == [] and dropped == {"score_mismatch": 1}
    out_of_range = _row("HIGH")
    out_of_range["ranking_score"] = 1.2
    dropped = {}
    assert _select([out_of_range], dropped) == [] and dropped == {"score_out_of_range": 1}


@pytest.mark.parametrize("extra, reason", [
    ({"days_to_earnings": 10.9}, "outside_entry_window"),
    ({"days_to_earnings": "9"}, "outside_entry_window"),
    ({"iv30": 0.0, "atm_iv": 0.35}, "implausible_or_missing_atm_iv"),
    ({"earnings_source_stale": 1}, "stale_earnings_date"),
    ({"earnings_source_stale": "true"}, "stale_earnings_date"),
    ({"earnings_source_stale": "yes"}, "stale_earnings_date"),
])
def test_no_permissive_coercion(extra, reason):
    dropped = {}
    assert _select([_row("X", **extra)], dropped) == []
    assert dropped == {reason: 1}


@pytest.mark.parametrize("value", [False, None, 0, "false", "0"])
def test_explicit_not_stale_values_pass(value):
    assert [r["symbol"] for r in _select([_row("FRESH", earnings_source_stale=value)])] == ["FRESH"]


@pytest.mark.parametrize("built_at, phase", [
    (datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc), "Built 08:00 EDT (before the US open)"),
    (datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc), "Built 11:00 EDT (after the US open)"),
    (datetime(2026, 10, 12, 20, 30, tzinfo=timezone.utc), "Built 16:30 EDT (after the US close)"),
    (datetime(2026, 10, 10, 15, 0, tzinfo=timezone.utc), "Built 11:00 EDT (US market closed today)"),
])
def test_market_phase_wording_follows_the_clock(built_at, phase):
    msg = format_alert_message([_row("TSLA")], as_of=built_at.date(), min_score=0.65, built_at=built_at)
    assert phase in msg


def _send(tmp_path, audit, rows, **patches):
    state = tmp_path / "s.json"
    with patch("scripts.premarket_screener_alert.build_ranked_screener", return_value=_table(rows)), \
            patch("scripts.premarket_screener_alert.IMessageConfig.from_env", return_value=object()), \
            patch("scripts.premarket_screener_alert.send_imessage",
                  **(patches.pop("send") if "send" in patches else {"return_value": {"to": "***1"}})) as send:
        code = main(["--state-path", str(state), "--audit-path", str(audit)])
    return code, send, state


def test_an_unwritable_audit_blocks_the_send(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    code, send, state = _send(tmp_path, blocker / "audit.jsonl", [_row("TSLA")])
    assert code == 1
    send.assert_not_called()
    assert not state.exists()


def test_a_failed_send_is_recorded_and_not_marked_sent(tmp_path):
    audit = tmp_path / "audit.jsonl"
    code, send, state = _send(tmp_path, audit, [_row("TSLA")], send={"side_effect": RuntimeError("osascript")})
    assert code == 1 and send.call_count == 1 and not state.exists()
    statuses = [json.loads(line)["status"] for line in audit.read_text().splitlines()]
    assert statuses == ["prepared", "failed"]


def test_a_sent_alert_whose_closing_line_fails_exits_nonzero_but_is_not_resent(tmp_path):
    audit = tmp_path / "audit.jsonl"
    with patch("scripts.premarket_screener_alert._try_append_audit", return_value=False):
        code, send, state = _send(tmp_path, audit, [_row("TSLA")])
    assert code == 1 and send.call_count == 1
    assert json.loads(state.read_text())["symbols"] == ["TSLA"]       # digest saved: no double send
    (pending,) = [json.loads(line) for line in audit.read_text().splitlines()]
    assert pending["status"] == "prepared" and pending["setups"][0]["row"]["symbol"] == "TSLA"


def test_non_finite_values_in_ungated_fields_do_not_block_the_record(tmp_path):
    audit = tmp_path / "audit.jsonl"
    code, send, _ = _send(tmp_path, audit, [_row("TSLA", rv30=float("nan"))])
    assert code == 0 and send.call_count == 1
    record = json.loads(audit.read_text().splitlines()[0])
    assert record["setups"][0]["row"]["rv30"] is None
