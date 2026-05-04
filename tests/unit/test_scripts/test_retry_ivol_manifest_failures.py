import json
from pathlib import Path

from scripts.retry_ivol_manifest_failures import load_retry_plans


def test_load_retry_plans_selects_failed_entries_and_preserves_request_shape(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "symbol": "ZTS",
                        "trade_date": "2025-05-07",
                        "endpoint": "/equities/eod/stock-opts-by-param",
                        "request_kind": "calls_dte7_14",
                        "params": {"symbol": "ZTS", "tradeDate": "2025-05-07", "cp": "C"},
                        "output_path": "/tmp/zts.json",
                        "expected_market_open": True,
                        "status": "failed",
                    },
                    {
                        "symbol": "AIZ",
                        "trade_date": "2025-07-22",
                        "endpoint": "/equities/eod/stock-opts-by-param",
                        "request_kind": "calls_dte45_90",
                        "params": {"symbol": "AIZ", "tradeDate": "2025-07-22", "cp": "C"},
                        "output_path": "/tmp/aiz.json",
                        "expected_market_open": True,
                        "status": "completed",
                    },
                ]
            }
        )
    )

    plans = load_retry_plans(manifest_path)

    assert len(plans) == 1
    assert plans[0].symbol == "ZTS"
    assert plans[0].trade_date == "2025-05-07"
    assert plans[0].request_kind == "calls_dte7_14"
    assert plans[0].params["tradeDate"] == "2025-05-07"
