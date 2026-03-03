import numpy as np

from scripts.institutional_backfill import InstitutionalDataCollector


def test_forward_quality_gate_passes_with_healthy_metrics():
    summary = {
        "status": "ok",
        "max_drawdown": -1200.0,
        "mean_execution_drag_bps": 95.0,
        "crush_directional_accuracy": 0.72,
        "fill_metrics": {"fill_events": 12},
    }
    passed, details, advisories = InstitutionalDataCollector._evaluate_forward_quality_gate(summary)
    assert passed is True
    assert details["status_ok"] is True
    assert details["drawdown_ok"] is True
    assert details["execution_drag_ok"] is True
    assert details["directional_accuracy_ok"] is True
    assert details["fill_log_ok"] is True
    assert advisories == []


def test_forward_quality_gate_surfaces_all_failures():
    summary = {
        "status": "insufficient_data",
        "max_drawdown": -100_000.0,
        "mean_execution_drag_bps": 320.0,
        "crush_directional_accuracy": 0.41,
        "fill_metrics": {"fill_events": 0},
    }
    passed, details, advisories = InstitutionalDataCollector._evaluate_forward_quality_gate(
        summary,
        require_status_ok=True,
        max_drawdown=-5_000.0,
        max_execution_drag_bps=200.0,
        min_directional_accuracy=0.55,
        require_fill_log=True,
    )
    assert passed is False
    assert details["status_ok"] is False
    assert details["drawdown_ok"] is False
    assert details["execution_drag_ok"] is False
    assert details["directional_accuracy_ok"] is False
    assert details["fill_log_ok"] is False
    assert len(advisories) >= 5


def test_forward_quality_gate_respects_disabled_thresholds():
    summary = {
        "status": "insufficient_data",
        "max_drawdown": np.nan,
        "mean_execution_drag_bps": np.nan,
        "crush_directional_accuracy": np.nan,
        "fill_metrics": {"fill_events": 0},
    }
    passed, details, advisories = InstitutionalDataCollector._evaluate_forward_quality_gate(
        summary,
        require_status_ok=False,
        max_drawdown=None,
        max_execution_drag_bps=None,
        min_directional_accuracy=None,
        require_fill_log=False,
    )
    assert passed is True
    assert details["status_ok"] is True
    assert details["drawdown_ok"] is True
    assert details["execution_drag_ok"] is True
    assert details["directional_accuracy_ok"] is True
    assert details["fill_log_ok"] is True
    assert advisories == []
