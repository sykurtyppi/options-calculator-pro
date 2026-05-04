import json
from datetime import date

from scripts.ivol_options_backfill import (
    BucketSpec,
    _build_bucket_specs,
    _month_partition,
    _response_summary,
    build_request_plan,
)


def test_build_bucket_specs_creates_both_sides_across_dte_ranges():
    specs = _build_bucket_specs(
        dte_buckets=[(7, 14), (20, 45)],
        call_put_values=["C", "P"],
        moneyness_from=-10,
        moneyness_to=10,
    )
    assert specs == [
        BucketSpec(7, 14, "C", -10, 10),
        BucketSpec(20, 45, "C", -10, 10),
        BucketSpec(7, 14, "P", -10, 10),
        BucketSpec(20, 45, "P", -10, 10),
    ]


def test_build_request_plan_adds_underlying_and_weekday_option_requests(tmp_path):
    specs = [BucketSpec(7, 14, "C", -10, 10), BucketSpec(20, 45, "P", -10, 10)]
    plans = build_request_plan(
        symbols=["SPY"],
        start_date=date(2025, 1, 3),
        end_date=date(2025, 1, 6),
        data_root=tmp_path,
        include_underlying=True,
        bucket_specs=specs,
    )

    assert plans[0].request_kind == "underlying_prices"
    option_plans = plans[1:]
    assert len(option_plans) == 4
    assert {plan.trade_date for plan in option_plans} == {"2025-01-03", "2025-01-06"}
    assert option_plans[0].params["symbol"] == "SPY"
    assert all(plan.expected_market_open is True for plan in option_plans)


def test_build_request_plan_marks_market_closed_dates_when_calendar_is_supplied(tmp_path):
    specs = [BucketSpec(7, 14, "C", -10, 10)]
    plans = build_request_plan(
        symbols=["SPY"],
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
        data_root=tmp_path,
        include_underlying=False,
        bucket_specs=specs,
        open_market_dates={date(2025, 1, 2)},
    )

    by_day = {plan.trade_date: plan for plan in plans}
    assert by_day["2025-01-01"].expected_market_open is False
    assert by_day["2025-01-02"].expected_market_open is True


def test_response_summary_reads_success_payload(tmp_path):
    payload = {
        "status": {"recordsFound": 12, "code": "COMPLETE"},
        "data": [{"id": 1}],
    }
    path = tmp_path / "response.json"
    path.write_text(json.dumps(payload))

    summary = _response_summary(path)
    assert summary["status_code"] == "COMPLETE"
    assert summary["records_found"] == 12
    assert summary["message"] is None


def test_month_partition_uses_utc_year_month_shape(tmp_path):
    partition = _month_partition(tmp_path)
    assert partition.parent == tmp_path / partition.parts[-2]
    assert len(partition.parts[-2]) == 4
    assert len(partition.parts[-1]) == 2
