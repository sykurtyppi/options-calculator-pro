"""Single source of truth for the crush-classifier feature transform.

Training (``institutional_ml_db.train_ml_model_on_historical_spreads``) and
serving (``edge_engine._ml_crush_probability`` and ``forward_screener``) MUST
produce identical feature vectors — same clip bounds, same fallbacks, same
order — or the served model sees z-scores it was never scaled or trained on
and ``predict_proba`` saturates. Historically the clips lived only in training,
so serving fed raw values straight into the scaler; this module removes that
train/serve skew by owning the transform for every caller.

Feature order (must match the fitted scaler/classifier):
    [near_back_ratio, log_front_iv, iv_rv_approx]
"""
from __future__ import annotations

import numpy as np

# Clip bounds applied at TRAIN time — serving applies the identical clips.
NBR_CLIP = (0.50, 4.0)
IV_RV_CLIP = (0.50, 5.0)
IV_FLOOR = 0.01          # front-IV floor before the log (and the NBR denominator floor)

# When realized vol is unavailable, training fills RV = front_iv * RV_FALLBACK_FACTOR,
# so the iv_rv RATIO becomes front_iv / (front_iv * RV_FALLBACK_FACTOR) = 1 / factor.
# Serving must use this SAME ratio for the no-RV case (the forward screener has no
# RV series), not `front_iv * factor`, which is a different quantity entirely.
RV_FALLBACK_FACTOR = 0.75
NO_RV_IV_RV_RATIO = 1.0 / RV_FALLBACK_FACTOR   # ≈ 1.3333
RV_FLOOR = 0.05         # realized-vol floor before forming the iv_rv ratio (train only)

# Neutral fallback for a non-finite NBR at serve time (pre-clip).
NBR_FALLBACK = 1.10

FEATURE_ORDER = ("near_back_ratio", "log_front_iv", "iv_rv_approx")


def _finite(value, fallback):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return v if np.isfinite(v) else float(fallback)


def crush_feature_vector(near_iv, near_back_ratio, iv_rv):
    """Return ``[near_back_ratio, log_front_iv, iv_rv_approx]`` with the exact
    train-time clips and fallbacks applied. Scalars in, list of 3 floats out.

    - near_back_ratio : front-month IV / back-month IV (already a ratio)
    - near_iv         : front-month ATM IV (drives log_front_iv)
    - iv_rv           : IV/RV ratio; pass ``NO_RV_IV_RV_RATIO`` when RV is absent
    """
    nbr = float(np.clip(_finite(near_back_ratio, NBR_FALLBACK), *NBR_CLIP))
    log_front_iv = float(np.log(max(_finite(near_iv, IV_FLOOR), IV_FLOOR)))
    ratio = float(np.clip(_finite(iv_rv, NO_RV_IV_RV_RATIO), *IV_RV_CLIP))
    return [nbr, log_front_iv, ratio]
