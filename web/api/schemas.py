from datetime import date, datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class EdgeAnalyzeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, description="Ticker symbol")


class EdgeAnalyzeResponse(BaseModel):
    generated_at: datetime
    symbol: str
    recommendation: str
    confidence_pct: float
    setup_score: float
    metrics: Dict[str, Any]
    rationale: list[str]
    selector_output: Optional[Dict[str, Any]] = None
    structure_scorecards: Optional[list[Dict[str, Any]]] = None
    vol_snapshot: Optional[Dict[str, Any]] = None


class OOSReportRequest(BaseModel):
    """OOS / backtest report parameters.

    Web-audit P2-1: every integer and float here previously had no upper
    bound. An authenticated user POSTing absurd values (``lookback_days=
    100_000_000``, ``max_backtest_symbols=100_000``) could pin a worker
    thread for the full 300s wall-clock and burn DuckDB/pandas memory.
    Bounds below are picked to cover any realistic backtest while
    refusing pathological inputs.
    """

    oos_stability_profile: Literal[
        "stability_auto",
        "evidence_balanced",
        "sample_expansion",
        "variance_control",
        "alpha_focus",
    ] = "stability_auto"

    # ~10y of history is enough for any realistic backtest.
    lookback_days: int = Field(1095, ge=30, le=3650)
    # 200 symbols is well above any production universe (10–50 typical).
    max_backtest_symbols: int = Field(50, ge=1, le=200)
    backtest_start_date: Optional[str] = "2023-01-01"
    backtest_end_date: Optional[str] = None
    min_signal_score: float = Field(0.50, ge=0.0, le=1.0)
    min_crush_confidence: float = Field(0.30, ge=0.0, le=1.0)
    min_crush_magnitude: float = Field(0.06, ge=0.0, le=1.0)
    min_crush_edge: float = Field(0.02, ge=0.0, le=1.0)
    target_entry_dte: int = Field(6, ge=0, le=365)
    entry_dte_band: int = Field(6, ge=0, le=365)
    min_daily_share_volume: int = Field(1_000_000, ge=0, le=10_000_000_000)
    max_abs_momentum_5d: float = Field(0.11, ge=0.0, le=10.0)

    oos_train_days: int = Field(189, ge=7, le=3650)
    oos_test_days: int = Field(42, ge=1, le=3650)
    oos_step_days: int = Field(42, ge=1, le=3650)
    oos_top_n_train: int = Field(1, ge=1, le=20)

    oos_min_splits: int = Field(8, ge=1, le=100)
    oos_min_total_test_trades: int = Field(80, ge=1, le=100_000)
    oos_min_trades_per_split: float = Field(5.0, ge=0.0, le=10_000.0)


class OOSReportResponse(BaseModel):
    generated_at: datetime
    summary: Dict[str, Any]
    output_files: Dict[str, Optional[str]]


class HistoricalOptionsResponse(BaseModel):
    generated_at: datetime
    symbol: str
    filters: Dict[str, Any]
    row_count: int
    rows: list[Dict[str, Any]]


class HistoricalOptionsCoverageResponse(BaseModel):
    generated_at: datetime
    coverage: list[Dict[str, Any]]


class ScreenerCheckResponse(BaseModel):
    label: str
    threshold: str
    actual: str
    passed: bool
    severity: Literal["hard", "soft"] = "hard"
    note: Optional[str] = None


class ScreenerRowResponse(BaseModel):
    symbol: str
    earnings_date: date
    release_timing: str
    entry_date: date
    entry_label: str
    selected_expiry: Optional[str] = None
    alternative_expiry: Optional[str] = None
    expiry_mode: Literal["front_after_earnings", "next_monthly_opex"]
    avg_spread_pct: Optional[float] = None
    previous_avg_spread_pct: Optional[float] = None
    spread_change_pct: Optional[float] = None
    spread_change_state: Literal["improved", "unchanged", "worsened", "new"] = "new"
    call_oi: Optional[int] = None
    put_oi: Optional[int] = None
    implied_move_pct: Optional[float] = None
    call_strike: Optional[float] = None
    put_strike: Optional[float] = None
    call_iv: Optional[float] = None
    put_iv: Optional[float] = None
    entry_debit_mid: Optional[float] = None
    status: Literal["QUALIFIED", "MARGINAL", "EXCLUDED"]
    status_reason: str
    compact_signal_summary: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    checks: list[ScreenerCheckResponse] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    last_updated: datetime
    detail_metrics: Dict[str, Any] = Field(default_factory=dict)


class EdgeScreenerResponse(BaseModel):
    generated_at: datetime
    expiry_mode: Literal["front_after_earnings", "next_monthly_opex"]
    as_of_date: date
    universe_size: int
    qualified_count: int
    marginal_count: int
    excluded_count: int
    rows: list[ScreenerRowResponse]


# ── Ranked screener (pre-earnings long-vega) ──────────────────────────────────

class RankedSetupRow(BaseModel):
    rank: int
    symbol: str
    earnings_date: Optional[date]
    dte: Optional[int]
    release_timing: str  # "BMO" | "AMC" | "UNKNOWN"
    iv_rv_ratio: Optional[float]
    atm_iv: Optional[float]
    rv30: Optional[float]
    ts_ratio: Optional[float]
    median_earnings_move_pct: Optional[float]
    sample_size: int
    spread_pct: Optional[float]
    ranking_score: Optional[float]
    score_components: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["ranked", "no_earnings", "error"]
    error_note: Optional[str] = None


class RankedScreenerResponse(BaseModel):
    generated_at: datetime
    as_of_date: date
    universe_size: int
    rows_returned: int
    in_entry_window: int
    ranking_weights: Dict[str, float]
    strategy_note: str
    rows: list[RankedSetupRow]


# ── Calibration ───────────────────────────────────────────────────────────────

class CalibrationBucket(BaseModel):
    score_lo: float
    score_hi: float
    score_mid: float
    expected_expansion_pct: float
    std_pct: float
    n: int
    prior_only: bool


class CalibrationCurveResponse(BaseModel):
    generated_at: datetime
    phase: Literal["bootstrap_prior", "observational", "fitted_moderate", "fitted_high"]
    n_observations: int
    min_for_observational: int
    min_for_fit: int
    min_for_high_fit: int
    buckets: list[CalibrationBucket]


class DistributionSummary(BaseModel):
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]


class LearningCalibrationDiagnostics(BaseModel):
    phase: Literal["bootstrap_prior", "observational", "fitted_moderate", "fitted_high"]
    n_total: int
    n_replay: int
    n_synthetic: int
    n_paper: int
    n_live: int
    is_prior_only: bool
    score_distribution: DistributionSummary
    expansion_distribution: DistributionSummary


class StructurePriorDiagnosticsEntry(BaseModel):
    count: int
    win_rate: float
    avg_return_pct: float
    avg_expansion_pct: float


class LearningDataQualityDiagnostics(BaseModel):
    has_real_data: bool
    replay_dominant: bool
    synthetic_ratio: float
    paper_ratio: float


class LearningHealthDiagnostics(BaseModel):
    calibration_stable: bool
    sufficient_observations: bool
    warning_flags: list[str]


class LearningDiagnosticsResponse(BaseModel):
    calibration: LearningCalibrationDiagnostics
    structure_priors: Dict[str, StructurePriorDiagnosticsEntry]
    data_quality: LearningDataQualityDiagnostics
    learning_health: LearningHealthDiagnostics
