from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math

import numpy as np
import pandas as pd

from services.option_surface_quality import diagnose_option_surface_quality
from services.realized_vol import (
    HAR_MIN_OBS as _HAR_MIN_OBS,
    RS_FALLBACK_WINDOW as _RS_FALLBACK_WINDOW,
    har_rv_forecast as _har_rv_forecast,
    rs_daily_vol_series as _rs_daily_vol_series,
    rs_trailing_mean_forecast as _rs_trailing_mean_forecast,
    yang_zhang_rv30 as _yang_zhang_rv30,
)


@dataclass(frozen=True)
class VolSnapshotConfig:
    max_term_expiries: int = 6
    max_smile_expiries: int = 3
    smile_max_abs_moneyness: float = 0.20
    smile_min_points: int = 5
    yz_window: int = 30
    regime_window_days: int = 252
    price_staleness_warn_days: int = 3
    chain_staleness_warn_days: int = 1
    exclude_event_contaminated_sessions: bool = True


@dataclass(frozen=True)
class VolSnapshot:
    # Identity / timing
    symbol: str
    as_of_date: date
    earnings_date: Optional[date]
    release_timing: str
    days_to_earnings: Optional[int]

    # Market state
    underlying_price: Optional[float]
    option_source: Optional[str]
    underlying_source: Optional[str]
    price_staleness_minutes: Optional[int]
    chain_staleness_minutes: Optional[int]
    data_quality: str
    data_quality_score: float

    # Realized-vol block
    rv30_yang_zhang: Optional[float]
    rv30_estimator: Optional[str]
    rv_har_forecast: Optional[float]
    rv_percentile_rank: Optional[float]
    vol_regime_label: str

    # Implied-vol block
    iv30: Optional[float]
    iv45: Optional[float]
    near_term_dte: Optional[int]
    near_term_atm_iv: Optional[float]
    back_term_dte: Optional[int]
    back_term_atm_iv: Optional[float]
    near_back_iv_ratio: Optional[float]
    term_structure_slope: Optional[float]

    # Event-pricing block
    near_term_implied_move_pct: Optional[float]
    near_term_implied_sigma_pct: Optional[float]
    non_event_move_pct_har: Optional[float]
    event_implied_move_pct: Optional[float]
    event_move_share_of_total: Optional[float]

    # Historical earnings behavior block
    historical_event_count: int
    historical_median_move_pct: Optional[float]
    historical_avg_last4_move_pct: Optional[float]
    historical_p90_move_pct: Optional[float]
    historical_move_std_pct: Optional[float]
    historical_move_anchor_pct: Optional[float]
    historical_move_uncertainty_pct: Optional[float]
    historical_vs_implied_move_ratio: Optional[float]
    tail_vs_implied_move_ratio: Optional[float]

    # Surface-shape block
    smile_curvature: Optional[float]
    smile_concavity_flag: Optional[bool]
    smile_points: int

    # Execution / liquidity block
    near_term_spread_pct: Optional[float]
    near_term_liquidity_proxy: Optional[float]
    atm_call_spread_pct: Optional[float]
    atm_put_spread_pct: Optional[float]
    atm_total_open_interest: Optional[float]
    atm_total_volume: Optional[float]
    liquidity_tier: str

    # Derived relationship block
    iv_rv_yz: Optional[float]
    iv_rv_har: Optional[float]
    cheapness_score: Optional[float]
    event_risk_score: Optional[float]
    execution_score: Optional[float]
    timing_score: Optional[float]

    # Audit / diagnostics
    historical_move_source: str
    null_reasons: Dict[str, str] = field(default_factory=dict)
    earnings_source_primary: Optional[str] = None
    earnings_source_confirmed: Optional[str] = None
    earnings_source_confidence: Optional[float] = None
    release_timing_source: Optional[str] = None
    earnings_source_stale: bool = False
    earnings_source_notes: List[str] = field(default_factory=list)
    surface_quality_status: Optional[str] = None
    surface_quality_reasons: List[str] = field(default_factory=list)
    surface_crossed_quote_count: int = 0
    surface_zero_bid_count: int = 0
    surface_extreme_spread_count: int = 0
    surface_sparse_atm_count: int = 0
    surface_iv_anomaly_count: int = 0
    surface_quality: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["as_of_date"] = self.as_of_date.isoformat()
        payload["earnings_date"] = self.earnings_date.isoformat() if self.earnings_date is not None else None
        return payload


@dataclass(frozen=True)
class _ResolvedEarnings:
    earnings_date: Optional[date]
    release_timing: str
    past_events: List[Dict[str, Any]]
    source_primary: Optional[str] = None
    source_confirmed: Optional[str] = None
    source_confidence: Optional[float] = None
    release_timing_source: Optional[str] = None
    source_stale: bool = False
    source_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class _HistoricalMoveProfile:
    earnings_event_count: int
    sample_size: int
    median_move_pct: Optional[float]
    avg_last4_move_pct: Optional[float]
    p90_move_pct: Optional[float]
    std_move_pct: Optional[float]
    source: str


@dataclass(frozen=True)
class _ExpiryATMStats:
    dte: int
    expiry: date
    atm_iv: Optional[float]
    call_spread_pct: Optional[float]
    put_spread_pct: Optional[float]
    total_open_interest: Optional[float]
    total_volume: Optional[float]
    implied_move_pct: Optional[float]
    liquidity_proxy: Optional[float]


@dataclass(frozen=True)
class _TermStructureSnapshot:
    iv30: Optional[float]
    iv45: Optional[float]
    near_term_dte: Optional[int]
    near_term_atm_iv: Optional[float]
    back_term_dte: Optional[int]
    back_term_atm_iv: Optional[float]
    near_back_iv_ratio: Optional[float]
    term_structure_slope: Optional[float]
    near_term_implied_move_pct: Optional[float]
    near_term_spread_pct: Optional[float]
    near_term_liquidity_proxy: Optional[float]
    atm_call_spread_pct: Optional[float]
    atm_put_spread_pct: Optional[float]
    atm_total_open_interest: Optional[float]
    atm_total_volume: Optional[float]
    point_count: int


def build_vol_snapshot(
    symbol: str,
    as_of: date | datetime,
    *,
    option_chain_data: Any = None,
    earnings_metadata: Any = None,
    price_data: Any = None,
    config: Optional[VolSnapshotConfig] = None,
) -> VolSnapshot:
    cfg = config or VolSnapshotConfig()
    as_of_date = _coerce_date(as_of)
    null_reasons: Dict[str, str] = {}

    price_frame, underlying_source = _normalize_price_frame(price_data)
    chain_frame, option_source = _normalize_option_chain(option_chain_data)
    earnings = _resolve_earnings_metadata(earnings_metadata, as_of_date)

    earnings_date = earnings.earnings_date
    days_to_earnings = (earnings_date - as_of_date).days if earnings_date is not None else None
    if earnings_date is None:
        null_reasons["earnings_date"] = "missing_earnings_metadata"
        null_reasons["days_to_earnings"] = "missing_earnings_date"

    underlying_price = _resolve_underlying_price(price_frame, chain_frame)
    if underlying_price is None:
        null_reasons["underlying_price"] = "missing_price_and_chain_underlying"

    surface_quality = diagnose_option_surface_quality(
        chain_frame,
        underlying_price=underlying_price,
        as_of_date=as_of_date,
    ).to_dict()

    price_staleness_minutes = _frame_staleness_minutes(price_frame, as_of_date)
    chain_staleness_minutes = _frame_staleness_minutes(chain_frame, as_of_date)
    if price_staleness_minutes is None:
        null_reasons["price_staleness_minutes"] = "missing_price_dates"
    if chain_staleness_minutes is None:
        null_reasons["chain_staleness_minutes"] = "missing_chain_trade_dates"

    rv30_yz: Optional[float] = None
    rv30_estimator: Optional[str] = None
    rv_har_forecast: Optional[float] = None
    rv_percentile_rank: Optional[float] = None
    vol_regime_label = "unknown"

    excluded_sessions: set[pd.Timestamp] = set()
    if cfg.exclude_event_contaminated_sessions and not price_frame.empty:
        excluded_sessions = _event_contaminated_session_dates(price_frame, earnings.past_events)

    baseline_price_frame = price_frame
    if excluded_sessions:
        baseline_price_frame = baseline_price_frame[
            [pd.Timestamp(ts).normalize() not in excluded_sessions for ts in baseline_price_frame.index]
        ]

    if not baseline_price_frame.empty and {"Open", "High", "Low", "Close"}.issubset(baseline_price_frame.columns):
        rv30_val = _yang_zhang_rv30(
            price_frame[["Open", "High", "Low", "Close"]],
            window=cfg.yz_window,
            excluded_sessions=excluded_sessions,
        )
        if np.isfinite(rv30_val) and rv30_val > 0:
            rv30_yz = float(rv30_val)
            rv30_estimator = "yang_zhang"
        else:
            if len(baseline_price_frame) < len(price_frame):
                null_reasons["rv30_yang_zhang"] = "insufficient_clean_ohlc_history_after_event_exclusion"
            else:
                null_reasons["rv30_yang_zhang"] = "insufficient_ohlc_history_for_yang_zhang"
            null_reasons["rv30_estimator"] = "rv30_yang_zhang_unavailable"

        rs_daily = _rs_daily_vol_series(
            price_frame[["Open", "High", "Low", "Close"]],
            excluded_sessions=excluded_sessions,
        )
        rv_har_val = _har_rv_forecast(rs_daily)
        if rv_har_val is not None and np.isfinite(rv_har_val) and rv_har_val > 0:
            rv_har_forecast = float(rv_har_val)
        else:
            # HAR requires n >= _HAR_MIN_OBS for stable OLS.  Try the simpler
            # RS trailing mean when the series is shorter but still long enough
            # for a reliable mean estimate.
            fallback_val = _rs_trailing_mean_forecast(rs_daily)
            if fallback_val is not None:
                rv_har_forecast = float(fallback_val)
                null_reasons["rv_har_estimator"] = (
                    f"rs_trailing_mean_fallback_n_lt_{_HAR_MIN_OBS}"
                )
            else:
                null_reasons["rv_har_forecast"] = (
                    "insufficient_clean_rs_history_after_event_exclusion"
                    if excluded_sessions
                    else "insufficient_rs_history_for_har"
                )

        if rv30_yz is not None:
            rv_pct, regime = _rv_percentile_and_regime(
                price_frame[["Open", "High", "Low", "Close"]],
                rv30_yz,
                window_days=cfg.regime_window_days,
                excluded_sessions=excluded_sessions,
            )
            rv_percentile_rank = rv_pct
            vol_regime_label = regime
            if rv_percentile_rank is None:
                null_reasons["rv_percentile_rank"] = "insufficient_history_for_regime_percentile"
    else:
        null_reasons["rv30_yang_zhang"] = "missing_ohlc_price_data"
        null_reasons["rv30_estimator"] = "missing_ohlc_price_data"
        null_reasons["rv_har_forecast"] = "missing_ohlc_price_data"
        null_reasons["rv_percentile_rank"] = "missing_ohlc_price_data"

    term = _build_term_structure_snapshot(chain_frame, underlying_price, as_of_date, cfg)
    if term.iv30 is None:
        null_reasons["iv30"] = "insufficient_term_structure_points"
    if term.iv45 is None:
        null_reasons["iv45"] = "insufficient_term_structure_points"
    if term.term_structure_slope is None:
        null_reasons["term_structure_slope"] = "insufficient_term_structure_points"
    if term.near_back_iv_ratio is None:
        null_reasons["near_back_iv_ratio"] = "insufficient_expiry_pairs"
    if term.near_term_implied_move_pct is None:
        null_reasons["near_term_implied_move_pct"] = "missing_atm_call_put_pair"

    # ── Event-vol decomposition (P-5a, variance-additive on calendar time) ──
    #
    # `near_term_implied_move_pct` is the legacy ATM straddle premium proxy
    # (call_mid + put_mid)/S × 100 — the mean-absolute-deviation (MAD) of
    # S_T/S_0 under risk-neutral lognormal (Brenner-Subrahmanyam 1988).
    # It is NOT a 1σ standard deviation, so it cannot be directly compared
    # to σ-form realized vol via quadrature subtraction.
    #
    # Convert to a 1σ-form percent move over [0, T] using the closed-form
    # identity:
    #     (call+put)/S = σ·√(2·T_years/π)
    #   ⇒ σ·√T_years   = (call+put)/S · √(π/2)
    #   ⇒ near_term_implied_sigma_pct = near_term_implied_move_pct · √(π/2)
    #
    # The conversion folds out T cleanly — the new field is dimensionally
    # 1σ-equivalent regardless of dte.
    #
    # Then perform a calendar-time variance decomposition (assume the event
    # consumes one calendar day of total variance; both σ_implied and σ_HAR
    # are quoted as annualised vols on calendar time per industry convention):
    #     σ_imp²·T = σ_HAR²·(T-1d) + σ_event²·1d
    #   ⇒ event_move²_pct = sigma²_pct − non_event_move²_pct
    #   where both terms are 1σ-form on calendar time.
    near_term_implied_sigma_pct: Optional[float] = None
    non_event_move_pct_har: Optional[float] = None
    event_implied_move_pct: Optional[float] = None
    event_move_share_of_total: Optional[float] = None
    if term.near_term_implied_move_pct is not None:
        near_term_implied_sigma_pct = float(
            term.near_term_implied_move_pct * math.sqrt(math.pi / 2.0)
        )
    if (
        near_term_implied_sigma_pct is not None
        and term.near_term_dte is not None
        and rv_har_forecast is not None
        and rv_har_forecast > 0
    ):
        non_event_T_years = max(int(term.near_term_dte) - 1, 0) / 365.0
        non_event_move_pct_har = float(
            rv_har_forecast * math.sqrt(non_event_T_years) * 100.0
        )
        event_variance_pct_sq = max(
            (near_term_implied_sigma_pct ** 2) - (non_event_move_pct_har ** 2),
            0.0,
        )
        event_implied_move_pct = float(math.sqrt(event_variance_pct_sq))
        if near_term_implied_sigma_pct > 0:
            event_move_share_of_total = float(
                event_implied_move_pct / near_term_implied_sigma_pct
            )
    else:
        if term.near_term_implied_move_pct is None:
            null_reasons["near_term_implied_sigma_pct"] = "near_term_implied_move_unavailable"
            null_reasons["event_implied_move_pct"] = "near_term_implied_move_unavailable"
            null_reasons["event_move_share_of_total"] = "near_term_implied_move_unavailable"
        elif rv_har_forecast is None or rv_har_forecast <= 0:
            null_reasons["event_implied_move_pct"] = "rv_har_forecast_unavailable"
            null_reasons["event_move_share_of_total"] = "rv_har_forecast_unavailable"
        elif term.near_term_dte is None:
            null_reasons["event_implied_move_pct"] = "near_term_dte_unavailable"
            null_reasons["event_move_share_of_total"] = "near_term_dte_unavailable"

    move_profile = _historical_earnings_move_profile(
        close=price_frame.get("Close", pd.Series(dtype=float)),
        earnings_events=earnings.past_events,
        as_of_date=as_of_date,
    )
    historical_move_anchor_pct = _compute_move_anchor(
        move_profile.median_move_pct,
        move_profile.avg_last4_move_pct,
    )
    historical_move_uncertainty_pct = _compute_move_uncertainty_pct(
        move_std_pct=move_profile.std_move_pct,
        sample_size=move_profile.sample_size,
        move_source=move_profile.source,
    )
    if historical_move_anchor_pct is None:
        null_reasons["historical_move_anchor_pct"] = "insufficient_historical_move_data"
    if historical_move_uncertainty_pct is None:
        null_reasons["historical_move_uncertainty_pct"] = "insufficient_historical_move_data"

    historical_vs_implied_move_ratio: Optional[float] = None
    tail_vs_implied_move_ratio: Optional[float] = None
    if historical_move_anchor_pct is not None and event_implied_move_pct is not None and event_implied_move_pct > 0:
        historical_vs_implied_move_ratio = float(historical_move_anchor_pct / event_implied_move_pct)
    else:
        null_reasons["historical_vs_implied_move_ratio"] = "historical_move_anchor_or_event_implied_move_unavailable"
    if move_profile.p90_move_pct is not None and event_implied_move_pct is not None and event_implied_move_pct > 0:
        tail_vs_implied_move_ratio = float(move_profile.p90_move_pct / event_implied_move_pct)
    else:
        null_reasons["tail_vs_implied_move_ratio"] = "historical_p90_or_event_implied_move_unavailable"

    smile_curvature, smile_concavity_flag, smile_points = _smile_curvature_from_chain(
        chain_frame,
        underlying_price,
        as_of_date,
        cfg,
    )
    if smile_curvature is None:
        null_reasons["smile_curvature"] = "insufficient_smile_points"
        null_reasons["smile_concavity_flag"] = "insufficient_smile_points"

    iv_rv_yz = float(term.iv30 / rv30_yz) if term.iv30 is not None and rv30_yz is not None and rv30_yz > 0 else None
    iv_rv_har = float(term.iv30 / rv_har_forecast) if term.iv30 is not None and rv_har_forecast is not None and rv_har_forecast > 0 else None
    if iv_rv_yz is None:
        null_reasons["iv_rv_yz"] = "iv30_or_rv30_yang_zhang_unavailable"
    if iv_rv_har is None:
        null_reasons["iv_rv_har"] = "iv30_or_rv_har_unavailable"

    # Higher cheapness_score means IV is lower relative to realized-vol baselines.
    cheapness_score = _cheapness_score(iv_rv_yz, iv_rv_har)
    # Higher event_risk_score means the option market is pricing a more event-dominated setup
    # and/or the stock's historical earnings moves have exceeded today's event-implied move.
    event_risk_score = _event_risk_score(
        event_move_share_of_total,
        historical_vs_implied_move_ratio,
        tail_vs_implied_move_ratio,
    )
    # Higher execution_score means tighter spreads and stronger liquidity. It is not a trade verdict.
    execution_score = _execution_score(term.near_term_spread_pct, term.near_term_liquidity_proxy)
    # Higher timing_score means the event is closer and therefore more relevant to current surface state.
    timing_score = _timing_score(days_to_earnings)

    data_quality_score = _data_quality_score(
        price_staleness_minutes=price_staleness_minutes,
        chain_staleness_minutes=chain_staleness_minutes,
        earnings_date=earnings_date,
        earnings_source_confidence=earnings.source_confidence,
        earnings_source_stale=earnings.source_stale,
        term_point_count=term.point_count,
        rv30_yz=rv30_yz,
        rv_har_forecast=rv_har_forecast,
        historical_move_source=move_profile.source,
        historical_event_count=move_profile.earnings_event_count,
    )
    data_quality = _quality_label(data_quality_score)

    if move_profile.median_move_pct is None:
        null_reasons["historical_median_move_pct"] = "no_historical_earnings_or_daily_fallback_data"
    if move_profile.avg_last4_move_pct is None:
        null_reasons["historical_avg_last4_move_pct"] = "no_historical_earnings_or_daily_fallback_data"
    if move_profile.p90_move_pct is None:
        null_reasons["historical_p90_move_pct"] = "no_historical_earnings_or_daily_fallback_data"
    if move_profile.std_move_pct is None:
        null_reasons["historical_move_std_pct"] = "no_historical_earnings_or_daily_fallback_data"
    if term.near_term_spread_pct is None:
        null_reasons["near_term_spread_pct"] = "missing_atm_bid_ask_quotes"
    if term.near_term_liquidity_proxy is None:
        null_reasons["near_term_liquidity_proxy"] = "missing_atm_open_interest_and_volume"
    if term.atm_call_spread_pct is None:
        null_reasons["atm_call_spread_pct"] = "missing_atm_call_bid_ask"
    if term.atm_put_spread_pct is None:
        null_reasons["atm_put_spread_pct"] = "missing_atm_put_bid_ask"
    if term.atm_total_open_interest is None:
        null_reasons["atm_total_open_interest"] = "missing_atm_open_interest"
    if term.atm_total_volume is None:
        null_reasons["atm_total_volume"] = "missing_atm_volume"
    if cheapness_score is None:
        null_reasons["cheapness_score"] = "iv_rv_inputs_unavailable"
    if event_risk_score is None:
        null_reasons["event_risk_score"] = "event_pricing_or_history_inputs_unavailable"
    if execution_score is None:
        null_reasons["execution_score"] = "spread_or_liquidity_inputs_unavailable"
    if timing_score is None:
        null_reasons["timing_score"] = "earnings_date_unavailable"
    if earnings.source_primary is None:
        null_reasons["earnings_source_primary"] = "earnings_source_provenance_unavailable"
    if earnings.source_confidence is None:
        null_reasons["earnings_source_confidence"] = "earnings_source_confidence_unavailable"
    if earnings.source_stale:
        null_reasons["earnings_source_stale"] = "stale_earnings_cache_used_after_refresh_failure"
    if non_event_move_pct_har is None:
        null_reasons["non_event_move_pct_har"] = "rv_har_forecast_or_near_term_dte_unavailable"
    if event_move_share_of_total is None and "event_move_share_of_total" not in null_reasons:
        null_reasons["event_move_share_of_total"] = "event_implied_move_or_near_term_implied_move_unavailable"

    return VolSnapshot(
        symbol=str(symbol).upper(),
        as_of_date=as_of_date,
        earnings_date=earnings_date,
        release_timing=earnings.release_timing,
        days_to_earnings=days_to_earnings,
        underlying_price=underlying_price,
        option_source=option_source,
        underlying_source=underlying_source,
        price_staleness_minutes=price_staleness_minutes,
        chain_staleness_minutes=chain_staleness_minutes,
        data_quality=data_quality,
        data_quality_score=float(data_quality_score),
        rv30_yang_zhang=rv30_yz,
        rv30_estimator=rv30_estimator,
        rv_har_forecast=rv_har_forecast,
        rv_percentile_rank=rv_percentile_rank,
        vol_regime_label=vol_regime_label,
        iv30=term.iv30,
        iv45=term.iv45,
        near_term_dte=term.near_term_dte,
        near_term_atm_iv=term.near_term_atm_iv,
        back_term_dte=term.back_term_dte,
        back_term_atm_iv=term.back_term_atm_iv,
        near_back_iv_ratio=term.near_back_iv_ratio,
        term_structure_slope=term.term_structure_slope,
        near_term_implied_move_pct=term.near_term_implied_move_pct,
        near_term_implied_sigma_pct=near_term_implied_sigma_pct,
        non_event_move_pct_har=non_event_move_pct_har,
        event_implied_move_pct=event_implied_move_pct,
        event_move_share_of_total=event_move_share_of_total,
        historical_event_count=move_profile.earnings_event_count,
        historical_median_move_pct=move_profile.median_move_pct,
        historical_avg_last4_move_pct=move_profile.avg_last4_move_pct,
        historical_p90_move_pct=move_profile.p90_move_pct,
        historical_move_std_pct=move_profile.std_move_pct,
        historical_move_anchor_pct=historical_move_anchor_pct,
        historical_move_uncertainty_pct=historical_move_uncertainty_pct,
        historical_vs_implied_move_ratio=historical_vs_implied_move_ratio,
        tail_vs_implied_move_ratio=tail_vs_implied_move_ratio,
        smile_curvature=smile_curvature,
        smile_concavity_flag=smile_concavity_flag,
        smile_points=smile_points,
        near_term_spread_pct=term.near_term_spread_pct,
        near_term_liquidity_proxy=term.near_term_liquidity_proxy,
        atm_call_spread_pct=term.atm_call_spread_pct,
        atm_put_spread_pct=term.atm_put_spread_pct,
        atm_total_open_interest=term.atm_total_open_interest,
        atm_total_volume=term.atm_total_volume,
        liquidity_tier=_liquidity_tier(term.near_term_liquidity_proxy),
        iv_rv_yz=iv_rv_yz,
        iv_rv_har=iv_rv_har,
        cheapness_score=cheapness_score,
        event_risk_score=event_risk_score,
        execution_score=execution_score,
        timing_score=timing_score,
        historical_move_source=move_profile.source,
        null_reasons=null_reasons,
        earnings_source_primary=earnings.source_primary,
        earnings_source_confirmed=earnings.source_confirmed,
        earnings_source_confidence=earnings.source_confidence,
        release_timing_source=earnings.release_timing_source,
        earnings_source_stale=earnings.source_stale,
        earnings_source_notes=list(earnings.source_notes),
        surface_quality_status=surface_quality.get("status"),
        surface_quality_reasons=list(surface_quality.get("warning_flags") or []),
        surface_crossed_quote_count=int(surface_quality.get("crossed_quote_count") or 0),
        surface_zero_bid_count=int(surface_quality.get("zero_bid_count") or 0),
        surface_extreme_spread_count=int(surface_quality.get("extreme_spread_count") or 0),
        surface_sparse_atm_count=int(surface_quality.get("sparse_atm_expiration_count") or 0),
        surface_iv_anomaly_count=int(surface_quality.get("missing_iv_count") or 0) + int(surface_quality.get("iv_outlier_count") or 0),
        surface_quality=surface_quality,
    )


def _coerce_date(value: date | datetime | pd.Timestamp) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return float(default)
        parsed = float(value)
        if not np.isfinite(parsed):
            return float(default)
        return parsed
    except (TypeError, ValueError):
        return float(default)


def _normalize_release_timing(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if any((value.hour, value.minute, value.second, value.microsecond)):
            total_minutes = (value.hour * 60) + value.minute
            if total_minutes < (9 * 60 + 30):
                return "before market open"
            if total_minutes >= (16 * 60):
                return "after market close"
            return "during market hours"
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    if "before" in text or text in {"bmo", "pre", "am"}:
        return "before market open"
    if "after" in text or text in {"amc", "post", "pm"}:
        return "after market close"
    if "during" in text or "intraday" in text:
        return "during market hours"
    return "unknown"


def _normalize_price_frame(price_data: Any) -> Tuple[pd.DataFrame, Optional[str]]:
    if price_data is None:
        return pd.DataFrame(), None
    if isinstance(price_data, pd.Series):
        df = pd.DataFrame({"Close": pd.to_numeric(price_data, errors="coerce")})
        if isinstance(price_data.index, pd.DatetimeIndex):
            df.index = price_data.index
        return _finalize_price_frame(df), "provided"
    if not isinstance(price_data, pd.DataFrame):
        return pd.DataFrame(), None

    df = price_data.copy()
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    date_col = None
    for candidate in ("trade_date", "date", "datetime", "timestamp"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)

    return _finalize_price_frame(df), "provided"


def _finalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")]
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _normalize_option_chain(option_chain_data: Any) -> Tuple[pd.DataFrame, Optional[str]]:
    if option_chain_data is None or not isinstance(option_chain_data, pd.DataFrame) or option_chain_data.empty:
        return pd.DataFrame(), None

    df = option_chain_data.copy()

    col_aliases = {
        "expiration_date": "expiry",
        "report_date": "trade_date",
        "side": "call_put",
        "option_type": "call_put",
        "impliedVolatility": "iv",
        "implied_volatility": "iv",
        "openInterest": "open_interest",
        "liquidityScore": "liquidity_score",
        "underlyingPrice": "underlying_price",
        "lastPrice": "last_price",
    }
    for src, dst in col_aliases.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    if "trade_date" not in df.columns and "updated" in df.columns:
        updated = pd.to_datetime(df["updated"], unit="s", utc=True, errors="coerce")
        if updated.notna().any():
            df["trade_date"] = updated.dt.tz_localize(None)

    for col in ("trade_date", "expiry"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()

    if "call_put" in df.columns:
        normalized_side = df["call_put"].astype(str).str.strip().str.upper()
        normalized_side = normalized_side.replace({"CALL": "C", "PUT": "P"})
        df["call_put"] = normalized_side

    for col in ("strike", "bid", "ask", "mid", "iv", "open_interest", "volume", "underlying_price", "spread_pct", "last_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mid" not in df.columns:
        df["mid"] = np.nan
    if {"bid", "ask"}.issubset(df.columns):
        valid_ba = np.isfinite(df["bid"]) & np.isfinite(df["ask"]) & (df["ask"] >= df["bid"]) & (df["ask"] > 0)
        df.loc[valid_ba, "mid"] = df.loc[valid_ba, "mid"].fillna((df.loc[valid_ba, "bid"] + df.loc[valid_ba, "ask"]) / 2.0)
    if "last_price" in df.columns:
        df["mid"] = df["mid"].fillna(df["last_price"])

    if "spread_pct" not in df.columns:
        df["spread_pct"] = np.nan
    if {"bid", "ask", "mid"}.issubset(df.columns):
        valid_spread = np.isfinite(df["bid"]) & np.isfinite(df["ask"]) & np.isfinite(df["mid"]) & (df["ask"] >= df["bid"]) & (df["mid"] > 0)
        df.loc[valid_spread, "spread_pct"] = ((df.loc[valid_spread, "ask"] - df.loc[valid_spread, "bid"]) / df.loc[valid_spread, "mid"]) * 100.0

    required_cols = [
        "trade_date",
        "expiry",
        "call_put",
        "strike",
        "bid",
        "ask",
        "mid",
        "iv",
        "open_interest",
        "volume",
        "underlying_price",
        "spread_pct",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    out = df.loc[:, required_cols].copy()
    out = out[out["expiry"].notna()]
    out = out[out["strike"].notna()]
    if "call_put" in out.columns:
        out = out[out["call_put"].isin({"C", "P"})]
    out = out.sort_values(["expiry", "call_put", "strike"]).reset_index(drop=True)
    return out, "provided"


def _resolve_earnings_metadata(earnings_metadata: Any, as_of_date: date) -> _ResolvedEarnings:
    if earnings_metadata is None:
        return _ResolvedEarnings(earnings_date=None, release_timing="unknown", past_events=[])

    if isinstance(earnings_metadata, Mapping):
        earnings_date = _extract_event_date(earnings_metadata.get("earnings_date") or earnings_metadata.get("event_date"))
        release_timing = _normalize_release_timing(earnings_metadata.get("release_timing") or earnings_metadata.get("reportTime"))
        prior_events = earnings_metadata.get("prior_events") or earnings_metadata.get("past_events") or []
        past_events = _parse_past_events(prior_events, as_of_date)
        return _ResolvedEarnings(
            earnings_date=earnings_date,
            release_timing=release_timing,
            past_events=past_events,
            source_primary=earnings_metadata.get("earnings_source_primary") or earnings_metadata.get("primary_source"),
            source_confirmed=earnings_metadata.get("earnings_source_confirmed") or earnings_metadata.get("confirmed_source"),
            source_confidence=(
                float(conf)
                if np.isfinite(
                    conf := _safe_float(
                        earnings_metadata.get("earnings_source_confidence") or earnings_metadata.get("source_confidence"),
                        np.nan,
                    )
                )
                else None
            ),
            release_timing_source=earnings_metadata.get("release_timing_source"),
            source_stale=bool(
                earnings_metadata.get("earnings_source_stale")
                or earnings_metadata.get("source_stale")
                or earnings_metadata.get("stale_cache")
            ),
            source_notes=list(
                earnings_metadata.get("earnings_source_notes")
                or earnings_metadata.get("source_notes")
                or []
            ),
        )

    if isinstance(earnings_metadata, pd.DataFrame):
        rows = earnings_metadata.to_dict("records")
    elif isinstance(earnings_metadata, Sequence) and not isinstance(earnings_metadata, (str, bytes)):
        rows = list(earnings_metadata)
    else:
        return _ResolvedEarnings(earnings_date=None, release_timing="unknown", past_events=[])

    parsed: List[Tuple[date, str]] = []
    past_events: List[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, Mapping):
            event_date = _extract_event_date(item.get("earnings_date") or item.get("event_date") or item.get("report_date"))
            release_timing = _normalize_release_timing(item.get("release_timing") or item.get("reportTime"))
        else:
            event_date = _extract_event_date(item)
            release_timing = _normalize_release_timing(item)
        if event_date is None:
            continue
        parsed.append((event_date, release_timing))
        if event_date <= as_of_date:
            past_events.append({"event_date": pd.Timestamp(event_date), "release_timing": release_timing})

    future_events = [(dt, timing) for dt, timing in parsed if dt >= as_of_date]
    future_events.sort(key=lambda item: item[0])
    if future_events:
        earnings_date, release_timing = future_events[0]
    else:
        earnings_date, release_timing = None, "unknown"
    return _ResolvedEarnings(earnings_date=earnings_date, release_timing=release_timing, past_events=past_events)


def _extract_event_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _parse_past_events(items: Iterable[Any], as_of_date: date) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            event_date = _extract_event_date(item.get("earnings_date") or item.get("event_date") or item.get("report_date"))
            release_timing = _normalize_release_timing(item.get("release_timing") or item.get("reportTime"))
        else:
            event_date = _extract_event_date(item)
            release_timing = _normalize_release_timing(item)
        if event_date is not None and event_date <= as_of_date:
            events.append({"event_date": pd.Timestamp(event_date), "release_timing": release_timing})
    events.sort(key=lambda item: pd.Timestamp(item["event_date"]))
    return events


def _resolve_underlying_price(price_frame: pd.DataFrame, chain_frame: pd.DataFrame) -> Optional[float]:
    if not price_frame.empty and "Close" in price_frame.columns:
        close = _safe_float(price_frame["Close"].dropna().iloc[-1] if not price_frame["Close"].dropna().empty else np.nan, np.nan)
        if np.isfinite(close) and close > 0:
            return float(close)
    if not chain_frame.empty and "underlying_price" in chain_frame.columns:
        vals = pd.to_numeric(chain_frame["underlying_price"], errors="coerce").dropna()
        if not vals.empty:
            return float(vals.median())
    return None


def _frame_staleness_minutes(frame: pd.DataFrame, as_of_date: date) -> Optional[int]:
    if frame.empty:
        return None
    if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index) > 0:
        latest = frame.index.max()
    elif "trade_date" in frame.columns:
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
        if dates.empty:
            return None
        latest = dates.max()
    else:
        return None
    latest_date = pd.Timestamp(latest).date()
    return int((as_of_date - latest_date).days * 1440)


def _build_term_structure_snapshot(
    chain_frame: pd.DataFrame,
    underlying_price: Optional[float],
    as_of_date: date,
    cfg: VolSnapshotConfig,
) -> _TermStructureSnapshot:
    empty = _TermStructureSnapshot(
        iv30=None,
        iv45=None,
        near_term_dte=None,
        near_term_atm_iv=None,
        back_term_dte=None,
        back_term_atm_iv=None,
        near_back_iv_ratio=None,
        term_structure_slope=None,
        near_term_implied_move_pct=None,
        near_term_spread_pct=None,
        near_term_liquidity_proxy=None,
        atm_call_spread_pct=None,
        atm_put_spread_pct=None,
        atm_total_open_interest=None,
        atm_total_volume=None,
        point_count=0,
    )
    if chain_frame.empty or underlying_price is None or not np.isfinite(underlying_price):
        return empty

    expiry_stats: List[_ExpiryATMStats] = []
    expiries = sorted({pd.Timestamp(v).date() for v in chain_frame["expiry"].dropna().tolist() if pd.Timestamp(v).date() > as_of_date})
    for expiry in expiries[: cfg.max_term_expiries]:
        grp = chain_frame[pd.to_datetime(chain_frame["expiry"], errors="coerce").dt.date == expiry].copy()
        stats = _expiry_atm_stats(grp, underlying_price, as_of_date, expiry)
        if stats.atm_iv is not None:
            expiry_stats.append(stats)

    if not expiry_stats:
        return empty

    near = expiry_stats[0]
    if len(expiry_stats) < 2:
        return _TermStructureSnapshot(
            iv30=None,
            iv45=None,
            near_term_dte=near.dte,
            near_term_atm_iv=near.atm_iv,
            back_term_dte=None,
            back_term_atm_iv=None,
            near_back_iv_ratio=None,
            term_structure_slope=None,
            near_term_implied_move_pct=near.implied_move_pct,
            near_term_spread_pct=_mean_or_none([near.call_spread_pct, near.put_spread_pct]),
            near_term_liquidity_proxy=near.liquidity_proxy,
            atm_call_spread_pct=near.call_spread_pct,
            atm_put_spread_pct=near.put_spread_pct,
            atm_total_open_interest=near.total_open_interest,
            atm_total_volume=near.total_volume,
            point_count=1,
        )

    days_arr = np.array([float(row.dte) for row in expiry_stats], dtype=float)
    ivs_arr = np.array([float(row.atm_iv) for row in expiry_stats if row.atm_iv is not None], dtype=float)
    order = np.argsort(days_arr)
    days_arr = days_arr[order]
    ivs_arr = ivs_arr[order]
    ordered_stats = [expiry_stats[idx] for idx in order]

    iv30 = float(np.interp(30.0, days_arr, ivs_arr))
    iv45 = float(np.interp(45.0, days_arr, ivs_arr))
    near = ordered_stats[0]
    back = ordered_stats[1]
    far_idx = int(np.argmin(np.abs(days_arr - 45.0)))
    far_dte = float(days_arr[far_idx])
    term_structure_slope = float((ivs_arr[far_idx] - ivs_arr[0]) / max(far_dte - float(days_arr[0]), 1.0))
    near_back_iv_ratio = float(near.atm_iv / back.atm_iv) if near.atm_iv is not None and back.atm_iv is not None and back.atm_iv > 0 else None

    return _TermStructureSnapshot(
        iv30=iv30,
        iv45=iv45,
        near_term_dte=near.dte,
        near_term_atm_iv=near.atm_iv,
        back_term_dte=back.dte,
        back_term_atm_iv=back.atm_iv,
        near_back_iv_ratio=near_back_iv_ratio,
        term_structure_slope=term_structure_slope,
        near_term_implied_move_pct=near.implied_move_pct,
        near_term_spread_pct=_mean_or_none([near.call_spread_pct, near.put_spread_pct]),
        near_term_liquidity_proxy=near.liquidity_proxy,
        atm_call_spread_pct=near.call_spread_pct,
        atm_put_spread_pct=near.put_spread_pct,
        atm_total_open_interest=near.total_open_interest,
        atm_total_volume=near.total_volume,
        point_count=len(ordered_stats),
    )


def _expiry_atm_stats(
    frame: pd.DataFrame,
    underlying_price: float,
    as_of_date: date,
    expiry: date,
) -> _ExpiryATMStats:
    calls = frame[frame["call_put"] == "C"].copy()
    puts = frame[frame["call_put"] == "P"].copy()
    call_row, put_row = _select_atm_pair(calls, puts, underlying_price)

    if call_row is None:
        call_row = _nearest_strike_row(calls, underlying_price)
    if put_row is None:
        put_row = _nearest_strike_row(puts, underlying_price)

    iv_candidates = [
        _safe_float((row or {}).get("iv"), np.nan)
        for row in (call_row, put_row)
    ]
    iv_candidates = [val for val in iv_candidates if np.isfinite(val) and val > 0]
    atm_iv = float(np.mean(iv_candidates)) if iv_candidates else None

    call_mid = _safe_float((call_row or {}).get("mid"), np.nan)
    put_mid = _safe_float((put_row or {}).get("mid"), np.nan)
    implied_move_pct = None
    if np.isfinite(call_mid) and np.isfinite(put_mid) and underlying_price > 0:
        implied_move_pct = float(((call_mid + put_mid) / underlying_price) * 100.0)

    call_spread_pct = _finite_or_none(_safe_float((call_row or {}).get("spread_pct"), np.nan))
    put_spread_pct = _finite_or_none(_safe_float((put_row or {}).get("spread_pct"), np.nan))
    call_oi = _finite_or_none(_safe_float((call_row or {}).get("open_interest"), np.nan))
    put_oi = _finite_or_none(_safe_float((put_row or {}).get("open_interest"), np.nan))
    call_vol = _finite_or_none(_safe_float((call_row or {}).get("volume"), np.nan))
    put_vol = _finite_or_none(_safe_float((put_row or {}).get("volume"), np.nan))

    total_oi = None
    if call_oi is not None or put_oi is not None:
        total_oi = float(max(call_oi or 0.0, 0.0) + max(put_oi or 0.0, 0.0))
    total_volume = None
    if call_vol is not None or put_vol is not None:
        total_volume = float(max(call_vol or 0.0, 0.0) + max(put_vol or 0.0, 0.0))
    liquidity_proxy = None
    if total_oi is not None or total_volume is not None:
        liquidity_proxy = float(max(total_oi or 0.0, 0.0) + max(total_volume or 0.0, 0.0))

    return _ExpiryATMStats(
        dte=max((expiry - as_of_date).days, 0),
        expiry=expiry,
        atm_iv=atm_iv,
        call_spread_pct=call_spread_pct,
        put_spread_pct=put_spread_pct,
        total_open_interest=total_oi,
        total_volume=total_volume,
        implied_move_pct=implied_move_pct,
        liquidity_proxy=liquidity_proxy,
    )


def _select_atm_pair(calls: pd.DataFrame, puts: pd.DataFrame, underlying_price: float) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    if calls.empty or puts.empty:
        return None, None
    common_strikes = sorted(set(pd.to_numeric(calls["strike"], errors="coerce").dropna().tolist()) & set(pd.to_numeric(puts["strike"], errors="coerce").dropna().tolist()))
    if not common_strikes:
        return None, None
    strike = min(common_strikes, key=lambda val: abs(float(val) - underlying_price))
    call_row = _best_liquidity_row(calls[np.isclose(pd.to_numeric(calls["strike"], errors="coerce"), float(strike), atol=1e-8)])
    put_row = _best_liquidity_row(puts[np.isclose(pd.to_numeric(puts["strike"], errors="coerce"), float(strike), atol=1e-8)])
    return call_row, put_row


def _nearest_strike_row(frame: pd.DataFrame, underlying_price: float) -> Optional[Mapping[str, Any]]:
    if frame.empty:
        return None
    tmp = frame.copy()
    tmp["_strike_dist"] = (pd.to_numeric(tmp["strike"], errors="coerce") - float(underlying_price)).abs()
    tmp = tmp.sort_values(["_strike_dist"])
    if tmp.empty:
        return None
    nearest_strike = float(pd.to_numeric(tmp.iloc[0]["strike"], errors="coerce"))
    subset = tmp[np.isclose(pd.to_numeric(tmp["strike"], errors="coerce"), nearest_strike, atol=1e-8)]
    return _best_liquidity_row(subset)


def _best_liquidity_row(frame: pd.DataFrame) -> Optional[Mapping[str, Any]]:
    if frame.empty:
        return None
    ranked = frame.copy()
    ranked["_liq_oi"] = pd.to_numeric(ranked["open_interest"], errors="coerce").fillna(0.0)
    ranked["_liq_vol"] = pd.to_numeric(ranked["volume"], errors="coerce").fillna(0.0)
    ranked["_liq_spread"] = pd.to_numeric(ranked["spread_pct"], errors="coerce").fillna(np.inf)
    ranked = ranked.sort_values(["_liq_oi", "_liq_vol", "_liq_spread"], ascending=[False, False, True])
    return ranked.iloc[0].to_dict()


def _smile_curvature_from_chain(
    chain_frame: pd.DataFrame,
    underlying_price: Optional[float],
    as_of_date: date,
    cfg: VolSnapshotConfig,
) -> Tuple[Optional[float], Optional[bool], int]:
    if chain_frame.empty or underlying_price is None or not np.isfinite(underlying_price):
        return None, None, 0

    expiries = sorted({pd.Timestamp(v).date() for v in chain_frame["expiry"].dropna().tolist() if pd.Timestamp(v).date() > as_of_date})
    for expiry in expiries[: cfg.max_smile_expiries]:
        grp = chain_frame[pd.to_datetime(chain_frame["expiry"], errors="coerce").dt.date == expiry].copy()
        rows: List[Tuple[float, float, float]] = []
        for strike, strike_rows in grp.groupby("strike"):
            iv_values = pd.to_numeric(strike_rows["iv"], errors="coerce").dropna()
            if iv_values.empty:
                continue
            moneyness = float(float(strike) / max(underlying_price, 1e-6) - 1.0)
            if abs(moneyness) > cfg.smile_max_abs_moneyness:
                continue

            oi_val = pd.to_numeric(strike_rows["open_interest"], errors="coerce").dropna()
            spread_val = pd.to_numeric(strike_rows["spread_pct"], errors="coerce").dropna()
            if not oi_val.empty and float(oi_val.sum()) > 0:
                weight = float(np.sqrt(float(oi_val.sum())))
            elif not spread_val.empty:
                weight = float(1.0 / (1.0 + float(spread_val.mean())))
            else:
                weight = 1.0
            rows.append((moneyness, float(iv_values.mean()), weight))

        if len(rows) < cfg.smile_min_points:
            continue

        rows.sort(key=lambda item: item[0])
        x = np.array([row[0] for row in rows], dtype=float)
        y = np.array([row[1] for row in rows], dtype=float)
        w = np.array([row[2] for row in rows], dtype=float)
        w = w / (w.sum() / len(w)) if w.sum() > 0 else np.ones_like(w)
        try:
            a, _, _ = np.polyfit(x, y, 2, w=w)
        except Exception:
            continue
        curvature = float(a)
        return curvature, bool(curvature < -0.5), int(len(rows))

    return None, None, 0


def _historical_earnings_move_profile(
    close: pd.Series,
    earnings_events: List[Any],
    as_of_date: date,
) -> _HistoricalMoveProfile:
    empty = _HistoricalMoveProfile(
        earnings_event_count=0,
        sample_size=0,
        median_move_pct=None,
        avg_last4_move_pct=None,
        p90_move_pct=None,
        std_move_pct=None,
        source="none",
    )
    if close is None or close.empty:
        return empty

    price_series = close.copy()
    if isinstance(price_series.index, pd.DatetimeIndex):
        idx = price_series.index.tz_localize(None) if price_series.index.tz is not None else price_series.index
        price_series.index = idx.normalize()
    price_series = price_series[~price_series.index.duplicated(keep="last")].sort_index()
    index_arr = price_series.index.to_numpy()
    if len(index_arr) < 10:
        return empty

    parsed_events: Dict[pd.Timestamp, Dict[str, Any]] = {}
    cutoff = pd.Timestamp(as_of_date)
    for item in earnings_events or []:
        try:
            if isinstance(item, Mapping):
                event_ts_raw = item.get("event_date")
                timing = _normalize_release_timing(item.get("release_timing"))
            else:
                event_ts_raw = item
                timing = _normalize_release_timing(item)
            if event_ts_raw is None:
                continue
            event_ts = pd.Timestamp(event_ts_raw).normalize()
            if event_ts > cutoff:
                continue
            existing = parsed_events.get(event_ts)
            if existing is None or existing.get("release_timing") == "unknown":
                parsed_events[event_ts] = {"event_date": event_ts, "release_timing": timing}
        except Exception:
            continue

    past_events = [parsed_events[key] for key in sorted(parsed_events.keys())]
    event_moves: List[float] = []
    for event in past_events[-24:]:
        event_ts = pd.Timestamp(event["event_date"]).normalize()
        release_timing = _normalize_release_timing(event.get("release_timing"))
        event_loc = int(index_arr.searchsorted(event_ts.to_datetime64(), side="left"))
        if event_loc >= len(index_arr):
            continue
        matched_event_session = pd.Timestamp(index_arr[event_loc]).normalize() == event_ts

        if release_timing == "after market close":
            if matched_event_session:
                pre_loc = event_loc
                post_loc = event_loc + 1
            else:
                pre_loc = event_loc - 1
                post_loc = event_loc
        else:
            pre_loc = event_loc - 1
            post_loc = event_loc

        if pre_loc < 0 or post_loc < 0 or pre_loc >= len(index_arr) or post_loc >= len(index_arr):
            continue

        pre_px = _safe_float(price_series.iloc[pre_loc], np.nan)
        post_px = _safe_float(price_series.iloc[post_loc], np.nan)
        if not np.isfinite(pre_px) or not np.isfinite(post_px) or pre_px <= 0:
            continue
        move_pct = abs((post_px - pre_px) / pre_px) * 100.0
        if np.isfinite(move_pct):
            event_moves.append(float(move_pct))

    if event_moves:
        moves = np.array(event_moves, dtype=float)
        if moves.size >= 5:
            low_clip, high_clip = np.percentile(moves, [1.0, 99.0])
            moves = np.clip(moves, low_clip, high_clip)
        return _HistoricalMoveProfile(
            earnings_event_count=int(len(event_moves)),
            sample_size=int(moves.size),
            median_move_pct=float(np.median(moves)),
            avg_last4_move_pct=float(np.mean(moves[-4:])),
            p90_move_pct=float(np.percentile(moves, 90)),
            std_move_pct=float(np.std(moves, ddof=1)) if moves.size > 1 else 0.0,
            source="earnings_history",
        )

    daily_moves = price_series.pct_change().abs().dropna().tail(126).to_numpy(dtype=float) * 100.0
    if daily_moves.size == 0:
        return empty
    if daily_moves.size >= 5:
        low_clip, high_clip = np.percentile(daily_moves, [1.0, 99.0])
        daily_moves = np.clip(daily_moves, low_clip, high_clip)
    return _HistoricalMoveProfile(
        earnings_event_count=0,
        sample_size=int(daily_moves.size),
        median_move_pct=float(np.median(daily_moves)),
        avg_last4_move_pct=float(np.mean(daily_moves[-4:])),
        p90_move_pct=float(np.percentile(daily_moves, 90)),
        std_move_pct=float(np.std(daily_moves, ddof=1)) if daily_moves.size > 1 else 0.0,
        source="daily_fallback",
    )


def _compute_move_anchor(median_move_pct: Optional[float], avg_last4_move_pct: Optional[float]) -> Optional[float]:
    median_val = _safe_float(median_move_pct, np.nan)
    avg4_val = _safe_float(avg_last4_move_pct, np.nan)
    if np.isfinite(median_val) and np.isfinite(avg4_val):
        return float(0.65 * avg4_val + 0.35 * median_val)
    if np.isfinite(avg4_val):
        return float(avg4_val)
    if np.isfinite(median_val):
        return float(median_val)
    return None


def _compute_move_uncertainty_pct(
    move_std_pct: Optional[float],
    sample_size: int,
    move_source: str,
) -> Optional[float]:
    std_val = _safe_float(move_std_pct, np.nan)
    if not np.isfinite(std_val) or std_val <= 0 or sample_size <= 0:
        return None
    if move_source == "earnings_history":
        effective_n = float(sample_size)
    elif move_source == "daily_fallback":
        effective_n = float(np.clip(sample_size * 0.35, 1.0, 10.0))
    else:
        effective_n = float(np.clip(sample_size * 0.25, 1.0, 6.0))
    uncertainty = 1.28 * std_val / np.sqrt(max(effective_n, 1.0))
    return float(np.clip(uncertainty, 0.03, 6.0))


def _event_contaminated_session_dates(
    hist: pd.DataFrame,
    earnings_events: Sequence[Mapping[str, Any]] | None,
) -> set[pd.Timestamp]:
    """
    Remove sessions contaminated by prior earnings-event shocks from the realized-vol baseline.

    Policy:
      - after market close: exclude the next trading session
      - before market open / during market hours / unknown: exclude the matching event session

    This keeps Yang-Zhang and HAR focused on ordinary background volatility instead of
    allowing recent event jumps to pollute the non-event baseline.
    """
    if hist.empty or not earnings_events or not isinstance(hist.index, pd.DatetimeIndex):
        return set()

    index_list = [pd.Timestamp(ts).normalize() for ts in hist.index]
    drop_dates: set[pd.Timestamp] = set()

    for item in earnings_events:
        try:
            event_ts = pd.Timestamp(item.get("event_date")).normalize()
        except Exception:
            continue
        timing = _normalize_release_timing(item.get("release_timing"))
        loc = hist.index.searchsorted(event_ts, side="left")
        if timing == "after market close":
            if loc < len(hist.index) and pd.Timestamp(hist.index[loc]).normalize() == event_ts:
                contam_loc = loc + 1
            else:
                contam_loc = loc
        else:
            contam_loc = loc
            if contam_loc < len(hist.index) and pd.Timestamp(hist.index[contam_loc]).normalize() != event_ts:
                continue

        if 0 <= contam_loc < len(index_list):
            drop_dates.add(index_list[contam_loc])

    return drop_dates


def _rv_percentile_and_regime(
    hist: pd.DataFrame,
    current_rv30: float,
    window_days: int = 252,
    excluded_sessions: Optional[set[pd.Timestamp]] = None,
) -> Tuple[Optional[float], str]:
    rs = _rs_daily_vol_series(hist, excluded_sessions=excluded_sessions)
    if len(rs) < 60:
        return None, "unknown"
    rolling_rv30 = rs.rolling(window=30, min_periods=20).mean()
    hist_vals = rolling_rv30.dropna().tail(window_days).values
    if len(hist_vals) < 30 or not np.isfinite(current_rv30) or current_rv30 <= 0:
        return None, "unknown"
    pct_rank = float(np.mean(hist_vals <= current_rv30) * 100.0)
    if pct_rank >= 75:
        return round(pct_rank, 1), "High"
    if pct_rank >= 50:
        return round(pct_rank, 1), "Elevated"
    if pct_rank >= 25:
        return round(pct_rank, 1), "Normal"
    return round(pct_rank, 1), "Low"


def _cheapness_score(iv_rv_yz: Optional[float], iv_rv_har: Optional[float]) -> Optional[float]:
    ratios = [float(val) for val in (iv_rv_yz, iv_rv_har) if val is not None and np.isfinite(val) and val > 0]
    if not ratios:
        return None
    avg_ratio = float(np.mean(ratios))
    score = 1.0 - float(np.clip((avg_ratio - 0.80) / 0.90, 0.0, 1.0))
    return float(np.clip(score, 0.0, 1.0))


def _event_risk_score(
    event_move_share_of_total: Optional[float],
    historical_vs_implied_move_ratio: Optional[float],
    tail_vs_implied_move_ratio: Optional[float],
) -> Optional[float]:
    components: List[float] = []
    if event_move_share_of_total is not None and np.isfinite(event_move_share_of_total):
        components.append(float(np.clip(event_move_share_of_total, 0.0, 1.0)))
    if historical_vs_implied_move_ratio is not None and np.isfinite(historical_vs_implied_move_ratio):
        components.append(float(np.clip((historical_vs_implied_move_ratio - 0.5) / 1.0, 0.0, 1.0)))
    if tail_vs_implied_move_ratio is not None and np.isfinite(tail_vs_implied_move_ratio):
        components.append(float(np.clip((tail_vs_implied_move_ratio - 0.7) / 1.0, 0.0, 1.0)))
    if not components:
        return None
    return float(np.mean(components))


def _execution_score(near_term_spread_pct: Optional[float], near_term_liquidity_proxy: Optional[float]) -> Optional[float]:
    if near_term_spread_pct is None and near_term_liquidity_proxy is None:
        return None
    spread_score = None
    if near_term_spread_pct is not None and np.isfinite(near_term_spread_pct):
        spread_score = 1.0 - float(np.clip(near_term_spread_pct / 25.0, 0.0, 1.0))
    liquidity_score = None
    if near_term_liquidity_proxy is not None and np.isfinite(near_term_liquidity_proxy):
        liquidity_score = float(
            np.clip(
                (np.log1p(max(float(near_term_liquidity_proxy), 1.0)) - np.log1p(250.0))
                / (np.log1p(25_000.0) - np.log1p(250.0)),
                0.0,
                1.0,
            )
        )
    components = [val for val in (spread_score, liquidity_score) if val is not None]
    if not components:
        return None
    return float(np.mean(components))


def _timing_score(days_to_earnings: Optional[int]) -> Optional[float]:
    """
    Score the timing of the entry relative to the earnings event.

    The pre-earnings long-vega strategy performs best when entered 5–8 DTE:
      - Too early (>14 DTE): theta drag erodes the position before IV expansion.
      - Too late (<2 DTE): IV expansion is largely complete; crush risk is immediate.
    A Gaussian centered at 6 DTE (sigma=3.5) captures this empirical window and is
    consistent with screener_service._dte_score() which uses the same parameterisation.

    The old formula (1 − dte/30) produced score=1.0 at 0 DTE, directly contradicting
    the strategy's optimal entry window and rewarding the worst possible timing.
    """
    if days_to_earnings is None or days_to_earnings < 0:
        return None
    return float(np.clip(
        np.exp(-0.5 * ((float(days_to_earnings) - 6.0) / 3.5) ** 2),
        0.0, 1.0,
    ))


def _data_quality_score(
    *,
    price_staleness_minutes: Optional[int],
    chain_staleness_minutes: Optional[int],
    earnings_date: Optional[date],
    earnings_source_confidence: Optional[float],
    earnings_source_stale: bool,
    term_point_count: int,
    rv30_yz: Optional[float],
    rv_har_forecast: Optional[float],
    historical_move_source: str,
    historical_event_count: int,
) -> float:
    price_score = _freshness_score(price_staleness_minutes, same_day_cutoff=1_440, warn_cutoff=4_320)
    chain_score = _freshness_score(chain_staleness_minutes, same_day_cutoff=1_440, warn_cutoff=2_880)
    earnings_score = 1.0 if earnings_date is not None else 0.0
    if earnings_source_confidence is None or not np.isfinite(float(earnings_source_confidence)):
        source_score = 0.40 if earnings_date is not None else 0.0
    else:
        source_score = float(np.clip(earnings_source_confidence, 0.0, 1.0))
    if earnings_source_stale:
        source_score *= 0.40
    term_score = float(np.clip(term_point_count / 4.0, 0.0, 1.0))
    rv_score = float(np.mean([
        1.0 if rv30_yz is not None else 0.0,
        1.0 if rv_har_forecast is not None else 0.0,
    ]))
    if historical_move_source == "earnings_history":
        history_score = float(np.clip(historical_event_count / 8.0, 0.0, 1.0))
    elif historical_move_source == "daily_fallback":
        history_score = 0.35
    else:
        history_score = 0.0
    score = (
        0.18 * price_score
        + 0.18 * chain_score
        + 0.12 * earnings_score
        + 0.12 * source_score
        + 0.14 * term_score
        + 0.14 * rv_score
        + 0.12 * history_score
    )
    if earnings_source_stale:
        score = min(score, 0.72)
    return float(np.clip(score, 0.0, 1.0))


def _freshness_score(staleness_minutes: Optional[int], same_day_cutoff: int, warn_cutoff: int) -> float:
    if staleness_minutes is None:
        return 0.0
    if staleness_minutes <= same_day_cutoff:
        return 1.0
    if staleness_minutes <= warn_cutoff:
        return 0.65
    if staleness_minutes <= warn_cutoff * 2:
        return 0.35
    return 0.10


def _quality_label(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "moderate"
    if score >= 0.40:
        return "low"
    return "poor"


def _liquidity_tier(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value >= 10_000:
        return "high"
    if value >= 3_000:
        return "mid"
    return "low"


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(val) for val in values if val is not None and np.isfinite(val)]
    if not finite:
        return None
    return float(np.mean(finite))


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None
