#!/usr/bin/env python3
"""
iVolatility historical options backfill.

Professional-grade raw data collection for EOD underlying prices and
parameterized options snapshots. The script is designed to build a durable raw
warehouse on external storage while keeping a structured manifest for every run.

Example:
    python scripts/ivol_options_backfill.py \
        --symbols SPY \
        --start-date 2025-01-01 \
        --end-date 2025-03-31 \
        --include-underlying \
        --max-dates 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env")
except Exception:
    pass

from utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_DTE_BUCKETS: tuple[tuple[int, int], ...] = ((7, 14), (20, 45), (45, 90))
DEFAULT_CP: tuple[str, ...] = ("C", "P")
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_SLEEP_SECONDS = 0.25
_API_KEY_RE = re.compile(r"([?&]apiKey=)[^&\s]+")


class IVolatilityEntitlementError(RuntimeError):
    """Raised when the API key is valid enough to respond but lacks access."""


def _redact_api_key(value: str) -> str:
    return _API_KEY_RE.sub(r"\1<redacted>", value)


def _response_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict):
        return str(payload.get("message") or payload.get("name") or payload)[:300]
    return str(payload)[:300]


def _raise_for_status_safely(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    detail = _response_error_message(response)
    raise requests.HTTPError(
        f"{response.status_code} Client Error: {response.reason} for url: "
        f"{_redact_api_key(response.url)}; message={detail}",
        response=response,
    )


@dataclass(frozen=True)
class BucketSpec:
    dte_from: int
    dte_to: int
    call_put: str
    moneyness_from: float
    moneyness_to: float

    @property
    def label(self) -> str:
        side = "calls" if self.call_put == "C" else "puts"
        return f"{side}_dte{self.dte_from}_{self.dte_to}"


@dataclass(frozen=True)
class RequestPlan:
    symbol: str
    trade_date: str
    endpoint: str
    params: Dict[str, Any]
    output_path: Path
    request_kind: str
    expected_market_open: bool = True


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _build_existing_stems_index(dirs: Sequence[Path]) -> set[str]:
    """Scan output directories once and build a set of filename stems without the run timestamp.

    Filenames follow: ivol_{symbol}_{bucket}_{trade_date}_{timestamp}.json
    We strip the last underscore-segment (the timestamp) to get a stable key that
    matches across runs, then store all found keys in a set for O(1) lookups.
    """
    index: set[str] = set()
    for d in dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix == ".json":
                parts = p.stem.rsplit("_", 1)
                if len(parts) == 2:
                    index.add(parts[0])
    return index


def _require_api_key(explicit_value: Optional[str]) -> str:
    api_key = explicit_value or os.environ.get("IVOLATILITY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("IVOLATILITY_API_KEY is not configured. Set it in .env or pass --api-key.")
    return api_key


def _resolve_data_root(explicit_root: Optional[str]) -> Path:
    configured = explicit_root or os.environ.get("MARKET_DATA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("/Volumes/T9/market_data")


def _month_partition(base_dir: Path) -> Path:
    now = datetime.now(UTC)
    return base_dir / f"{now:%Y}" / f"{now:%m}"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(" ", "")


def _iter_business_days(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            yield current
        current = current.fromordinal(current.toordinal() + 1)


def _build_bucket_specs(
    dte_buckets: Sequence[tuple[int, int]],
    call_put_values: Sequence[str],
    moneyness_from: float,
    moneyness_to: float,
) -> List[BucketSpec]:
    specs: List[BucketSpec] = []
    for cp in call_put_values:
        if cp not in {"C", "P"}:
            raise ValueError(f"Unsupported call/put value: {cp}")
        for dte_from, dte_to in dte_buckets:
            if dte_from > dte_to:
                raise ValueError(f"Invalid DTE bucket {dte_from}-{dte_to}")
            specs.append(
                BucketSpec(
                    dte_from=dte_from,
                    dte_to=dte_to,
                    call_put=cp,
                    moneyness_from=moneyness_from,
                    moneyness_to=moneyness_to,
                )
            )
    return specs


def _build_underlying_plan(
    symbol: str,
    start_date: date,
    end_date: date,
    data_root: Path,
) -> RequestPlan:
    output_dir = _month_partition(data_root / "raw" / "ivolatility" / "reference")
    filename = (
        f"ivol_stock_prices_{_safe_symbol(symbol).lower()}_"
        f"{start_date.isoformat()}_{end_date.isoformat()}_{_utc_timestamp()}.json"
    )
    return RequestPlan(
        symbol=symbol,
        trade_date=end_date.isoformat(),
        endpoint="/equities/eod/stock-prices",
        params={
            "symbol": symbol,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
        },
        output_path=output_dir / filename,
        request_kind="underlying_prices",
        expected_market_open=True,
    )


def _build_option_plan(
    symbol: str,
    trade_date: date,
    bucket: BucketSpec,
    data_root: Path,
    expected_market_open: bool,
) -> RequestPlan:
    output_dir = _month_partition(data_root / "raw" / "ivolatility" / "options_chains")
    filename = (
        f"ivol_{_safe_symbol(symbol).lower()}_{bucket.label}_"
        f"{trade_date.isoformat()}_{_utc_timestamp()}.json"
    )
    return RequestPlan(
        symbol=symbol,
        trade_date=trade_date.isoformat(),
        endpoint="/equities/eod/stock-opts-by-param",
        params={
            "symbol": symbol,
            "tradeDate": trade_date.isoformat(),
            "dteFrom": bucket.dte_from,
            "dteTo": bucket.dte_to,
            "moneynessFrom": bucket.moneyness_from,
            "moneynessTo": bucket.moneyness_to,
            "cp": bucket.call_put,
        },
        output_path=output_dir / filename,
        request_kind=bucket.label,
        expected_market_open=expected_market_open,
    )


def build_request_plan(
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    data_root: Path,
    include_underlying: bool,
    bucket_specs: Sequence[BucketSpec],
    open_market_dates: Optional[set[date]] = None,
    max_dates: Optional[int] = None,
) -> List[RequestPlan]:
    plans: List[RequestPlan] = []
    trade_dates = list(_iter_business_days(start_date, end_date))
    if max_dates is not None:
        trade_dates = trade_dates[:max_dates]

    for symbol in symbols:
        if include_underlying:
            plans.append(_build_underlying_plan(symbol, start_date, end_date, data_root))
        for trade_dt in trade_dates:
            expected_market_open = open_market_dates is None or trade_dt in open_market_dates
            for bucket in bucket_specs:
                plans.append(
                    _build_option_plan(
                        symbol=symbol,
                        trade_date=trade_dt,
                        bucket=bucket,
                        data_root=data_root,
                        expected_market_open=expected_market_open,
                    )
                )
    return plans


def _response_summary(output_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(output_path.read_text())
    except json.JSONDecodeError:
        return {"status_code": "invalid_json", "records_found": None, "response_code": None}

    status = payload.get("status", {}) if isinstance(payload, dict) else {}
    return {
        "status_code": payload.get("code") or status.get("code"),
        "records_found": status.get("recordsFound"),
        "response_code": payload.get("code"),
        "message": payload.get("message"),
    }


class IVolatilityBackfillRunner:
    def __init__(
        self,
        api_key: str,
        data_root: Path,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        skip_existing: bool = True,
    ) -> None:
        self.api_key = api_key
        self.data_root = data_root
        self.timeout_seconds = timeout_seconds
        self.sleep_seconds = sleep_seconds
        self.skip_existing = skip_existing
        self.base_url = "https://restapi.ivolatility.com"
        self.session = requests.Session()

    def fetch_trading_calendar(self, start_date: date, end_date: date) -> Optional[set[date]]:
        response = self.session.get(
            self.base_url + "/equities/trading-calendar",
            params={
                "apiKey": self.api_key,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
            },
            timeout=self.timeout_seconds,
        )
        if response.status_code in {401, 403}:
            logger.warning(
                "Trading calendar unavailable for current iVolatility entitlement (HTTP %s). "
                "Continuing without market-closed skipping.",
                response.status_code,
            )
            return None

        _raise_for_status_safely(response)
        payload = response.json()
        data = payload.get("data", []) if isinstance(payload, dict) else []
        open_dates: set[date] = set()
        for item in data:
            raw_value = (
                item.get("date")
                or item.get("tradingDate")
                or item.get("trade_date")
                or item.get("marketDate")
            )
            if not raw_value:
                continue
            try:
                open_dates.add(date.fromisoformat(str(raw_value)[:10]))
            except ValueError:
                continue
        return open_dates

    def execute(self, plans: Sequence[RequestPlan], run_label: str, dry_run: bool = False) -> Dict[str, Any]:
        started_at = datetime.now(UTC).isoformat()
        manifest_dir = _month_partition(self.data_root / "manifests" / "downloads")
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{run_label}_{_utc_timestamp()}.json"

        # Build a set of already-fetched stems (symbol+bucket+trade_date, no timestamp)
        # by scanning output dirs once — avoids per-plan glob on a million-file directory.
        existing_stems: set[str] = set()
        if self.skip_existing:
            output_dirs = {p.output_path.parent for p in plans}
            existing_stems = _build_existing_stems_index(list(output_dirs))

        results: List[Dict[str, Any]] = []
        counts = {"completed": 0, "skipped": 0, "failed": 0, "dry_run": 0}
        anomalies = {"unexpected_zero_record_days": 0, "market_closed_days_skipped": 0}

        for plan_index, plan in enumerate(plans):
            record = {
                "symbol": plan.symbol,
                "trade_date": plan.trade_date,
                "endpoint": plan.endpoint,
                "request_kind": plan.request_kind,
                "params": plan.params,
                "output_path": str(plan.output_path),
                "expected_market_open": plan.expected_market_open,
            }

            if dry_run:
                record["status"] = "dry_run"
                counts["dry_run"] += 1
                results.append(record)
                continue

            if not plan.expected_market_open and plan.request_kind != "underlying_prices":
                record["status"] = "skipped_market_closed"
                counts["skipped"] += 1
                anomalies["market_closed_days_skipped"] += 1
                results.append(record)
                continue

            stem_key = plan.output_path.stem.rsplit("_", 1)[0]
            if self.skip_existing and stem_key in existing_stems:
                record["status"] = "skipped_existing"
                counts["skipped"] += 1
                results.append(record)
                continue

            try:
                self._execute_plan(plan)
                summary = _response_summary(plan.output_path)
                record["status"] = "completed"
                record["file_size_bytes"] = plan.output_path.stat().st_size
                record.update(summary)
                record["anomaly"] = None
                if (
                    plan.request_kind != "underlying_prices"
                    and plan.expected_market_open
                    and str(summary.get("status_code")) == "COMPLETE"
                    and int(summary.get("records_found") or 0) == 0
                ):
                    record["anomaly"] = "unexpected_zero_records_open_market"
                    anomalies["unexpected_zero_record_days"] += 1
                counts["completed"] += 1
                logger.info(
                    "Captured %s %s %s -> %s bytes records=%s status=%s",
                    plan.symbol,
                    plan.trade_date,
                    plan.request_kind,
                    record["file_size_bytes"],
                    summary.get("records_found"),
                    summary.get("status_code"),
                )
            except IVolatilityEntitlementError as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                counts["failed"] += 1
                logger.error(
                    "Failed %s %s %s: %s",
                    plan.symbol,
                    plan.trade_date,
                    plan.request_kind,
                    exc,
                )
                results.append(record)
                for skipped_plan in plans[plan_index + 1 :]:
                    results.append(
                        {
                            "symbol": skipped_plan.symbol,
                            "trade_date": skipped_plan.trade_date,
                            "endpoint": skipped_plan.endpoint,
                            "request_kind": skipped_plan.request_kind,
                            "params": skipped_plan.params,
                            "output_path": str(skipped_plan.output_path),
                            "expected_market_open": skipped_plan.expected_market_open,
                            "status": "skipped_entitlement_abort",
                        }
                    )
                    counts["skipped"] += 1
                logger.error("Aborting run after iVolatility entitlement failure.")
                break
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = _redact_api_key(str(exc))
                counts["failed"] += 1
                logger.error(
                    "Failed %s %s %s: %s",
                    plan.symbol,
                    plan.trade_date,
                    plan.request_kind,
                    record["error"],
                )
                results.append(record)
            else:
                results.append(record)
            time.sleep(self.sleep_seconds)

        manifest = {
            "run_label": run_label,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "data_root": str(self.data_root),
            "counts": counts,
            "anomalies": anomalies,
            "results": results,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info("Wrote manifest to %s", manifest_path)
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def _execute_plan(self, plan: RequestPlan) -> None:
        _ensure_parent(plan.output_path)
        params = {"apiKey": self.api_key, **plan.params}
        response = self.session.get(
            self.base_url + plan.endpoint,
            params=params,
            timeout=self.timeout_seconds,
        )
        if response.status_code in {401, 403}:
            detail = _response_error_message(response)
            raise IVolatilityEntitlementError(
                f"iVolatility entitlement denied (HTTP {response.status_code}) for "
                f"{plan.endpoint}: {detail}"
            )
        _raise_for_status_safely(response)
        plan.output_path.write_bytes(response.content)


def _parse_dte_buckets(values: Sequence[str]) -> List[tuple[int, int]]:
    buckets: List[tuple[int, int]] = []
    for value in values:
        left, right = value.split("-", 1)
        buckets.append((int(left), int(right)))
    return buckets


def _parse_call_puts(value: str) -> List[str]:
    normalized = value.strip().upper()
    if normalized == "ALL":
        return list(DEFAULT_CP)
    return [item.strip().upper() for item in normalized.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill iVolatility EOD options data.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. SPY or AAPL,MSFT")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD")
    parser.add_argument("--api-key", default=None, help="Optional iVolatility API key override")
    parser.add_argument("--data-root", default=None, help="Optional market data root override")
    parser.add_argument("--include-underlying", action="store_true", help="Include a stock-prices pull per symbol")
    parser.add_argument(
        "--dte-bucket",
        action="append",
        default=[],
        help="DTE bucket in start-end form, repeatable. Defaults to 7-14,20-45,45-90.",
    )
    parser.add_argument(
        "--call-put",
        default="ALL",
        help="C, P, C,P, or ALL (default).",
    )
    parser.add_argument("--moneyness-from", type=float, default=-10.0)
    parser.add_argument("--moneyness-to", type=float, default=10.0)
    parser.add_argument("--max-dates", type=int, default=None, help="Limit number of business dates for pilot runs")
    parser.add_argument("--run-label", default="ivol_options_backfill")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-download files even if they already exist")
    parser.add_argument(
        "--skip-market-closed",
        action="store_true",
        help="Use iVolatility trading calendar to skip market-closed business dates for options requests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    if end_date < start_date:
        parser.error("--end-date must be on or after --start-date")

    dte_buckets = _parse_dte_buckets(args.dte_bucket) if args.dte_bucket else list(DEFAULT_DTE_BUCKETS)
    call_put_values = _parse_call_puts(args.call_put)
    bucket_specs = _build_bucket_specs(
        dte_buckets=dte_buckets,
        call_put_values=call_put_values,
        moneyness_from=args.moneyness_from,
        moneyness_to=args.moneyness_to,
    )

    api_key = _require_api_key(args.api_key)
    data_root = _resolve_data_root(args.data_root)

    runner = IVolatilityBackfillRunner(
        api_key=api_key,
        data_root=data_root,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
        skip_existing=not args.no_skip_existing,
    )
    open_market_dates: Optional[set[date]] = None
    if args.skip_market_closed:
        open_market_dates = runner.fetch_trading_calendar(start_date=start_date, end_date=end_date)
        if open_market_dates is not None:
            logger.info(
                "Loaded %d open market dates from trading calendar for %s to %s",
                len(open_market_dates),
                start_date.isoformat(),
                end_date.isoformat(),
            )
        else:
            logger.info(
                "Proceeding without trading calendar for %s to %s",
                start_date.isoformat(),
                end_date.isoformat(),
            )
    plans = build_request_plan(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        data_root=data_root,
        include_underlying=args.include_underlying,
        bucket_specs=bucket_specs,
        open_market_dates=open_market_dates,
        max_dates=args.max_dates,
    )
    logger.info("Prepared %d requests for %d symbols", len(plans), len(symbols))
    manifest = runner.execute(plans=plans, run_label=args.run_label, dry_run=args.dry_run)
    logger.info("Run complete: %s", json.dumps(manifest["counts"], sort_keys=True))
    return 0 if manifest["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
