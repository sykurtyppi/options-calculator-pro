#!/usr/bin/env python3
"""CLI driver for InstitutionalMLDatabase.train_ml_model_on_historical_spreads.

The training method is currently only callable via the /api/ml/train FastAPI
endpoint — this script lets us invoke it directly without spinning up the API,
which is what we need for research + the first-ever model training run.

The trainer:
  - reads labels from earnings_iv_decay_labels (1,639 events available)
  - features: near_back_ratio, log_front_iv, iv_rv_approx (constant fallback if
    daily_prices is empty — current state)
  - label: binary crush_happened (front_iv_crush_pct < -0.10) — see methodology
    note below; this default threshold is severely degenerate on the actual data
  - algorithm: CalibratedClassifierCV(LogisticRegression, isotonic, cv=3-5)
  - saves crush_classifier.pkl / crush_scaler.pkl / crush_model_meta.json to
    ~/.options_calculator_pro/models/

The live engine's _ml_crush_probability picks up those .pkl files on next
import. They feed only the dormant _calibration_mult metric (PR #84) — no
production decision impact unless caps/multipliers are later wired.

Usage:
  python scripts/train_crush_model.py                  # train + write report
  python scripts/train_crush_model.py --dry-run        # check prereqs only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from services.institutional_ml_db import InstitutionalMLDatabase  # noqa: E402

MODEL_DIR = Path.home() / ".options_calculator_pro" / "models"
REPORT_DIR = Path.home() / ".options_calculator_pro" / "reports"


def check_prereqs() -> dict:
    """Verify training data + dependencies before invoking the trainer."""
    import sqlite3
    db_path = Path.home() / ".options_calculator_pro" / "institutional_ml.db"
    info: dict = {"db_path": str(db_path), "db_exists": db_path.exists()}
    if not db_path.exists():
        return info
    con = sqlite3.connect(db_path)
    info["labels_eligible"] = con.execute("""
        SELECT COUNT(*) FROM earnings_iv_decay_labels
        WHERE quality_score >= 0.40
          AND pre_front_iv IS NOT NULL AND pre_front_iv > 0.01
          AND pre_back_iv IS NOT NULL AND pre_back_iv > 0.01
          AND front_iv_crush_pct IS NOT NULL
    """).fetchone()[0]
    info["daily_prices_with_rv"] = con.execute(
        "SELECT COUNT(*) FROM daily_prices "
        "WHERE realized_vol_30d IS NOT NULL AND realized_vol_30d > 0"
    ).fetchone()[0]
    crush, no_crush = con.execute("""
        SELECT
          SUM(CASE WHEN front_iv_crush_pct < -0.10 THEN 1 ELSE 0 END),
          SUM(CASE WHEN front_iv_crush_pct >= -0.10 THEN 1 ELSE 0 END)
        FROM earnings_iv_decay_labels
        WHERE quality_score >= 0.40
          AND pre_front_iv > 0.01 AND pre_back_iv > 0.01
          AND front_iv_crush_pct IS NOT NULL
    """).fetchone()
    info["label_at_minus10"] = {
        "positive": int(crush or 0),
        "negative": int(no_crush or 0),
        "positive_rate": (crush / (crush + no_crush)) if (crush or no_crush) else 0.0,
    }
    con.close()
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the ML crush classifier.")
    ap.add_argument("--dry-run", action="store_true",
                    help="check prerequisites only; do not train")
    args = ap.parse_args()

    print("=== Crush model training driver ===\n")
    info = check_prereqs()
    print("Prerequisites:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    if not info.get("db_exists"):
        print("\nABORT: institutional_ml.db not found.")
        return 2
    if info.get("labels_eligible", 0) < 30:
        print(f"\nABORT: {info.get('labels_eligible')} eligible labels < 30 minimum.")
        return 2

    bal = info.get("label_at_minus10", {})
    pos_rate = bal.get("positive_rate", 0)
    if pos_rate > 0.95 or pos_rate < 0.05:
        print(f"\n⚠️  Label balance at -10% threshold is DEGENERATE: "
              f"{pos_rate*100:.1f}% positive. The trained model will likely "
              "collapse to a constant predictor.")
        print("    (For a meaningfully-balanced target, -40% threshold yields ~45/55.)")

    if args.dry_run:
        print("\n--dry-run: not training.")
        return 0

    print("\nInvoking train_ml_model_on_historical_spreads() ...")
    db = InstitutionalMLDatabase()
    result = db.train_ml_model_on_historical_spreads()

    print("\nTrainer result:")
    print(json.dumps(result, indent=2, default=str))

    if result.get("trained"):
        print(f"\nModel artifacts at: {MODEL_DIR}")
        for f in sorted(MODEL_DIR.glob("crush_*")):
            print(f"  {f.name} ({f.stat().st_size} bytes)")

        # Persist a run report
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = REPORT_DIR / f"crush_model_training_{ts}.json"
        report.write_text(json.dumps(
            {"prereqs": info, "trainer_result": result},
            indent=2, default=str,
        ))
        print(f"  report: {report}")
        return 0

    print("\n❌ training did not produce a model.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
