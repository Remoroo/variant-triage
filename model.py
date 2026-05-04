"""Model factory for variant-triage. EDITABLE.

The baseline is a single LightGBM booster — fast, single-thread-friendly,
strong on this kind of mixed categorical/continuous input. The agent
may swap in:

- XGBoost / CatBoost (all three usually give within ±0.005 AUC; the
  hyperparams dominate).
- A small MLP (2-3 hidden layers, 64-128 units) — often useful in
  blends with a tree model.
- A calibrated stack / blend of several bases.

Returned model must:
- expose `predict_proba(X)` or `predict(X)` returning a 1-D positive-
  class score,
- run single-threaded (n_jobs=1),
- produce ≤ 100 MB of on-disk artefact.
"""
from __future__ import annotations

from typing import Any


def build_model() -> Any:
    """Returns an unfitted LightGBM classifier configured for 1 thread."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=63,
        min_data_in_leaf=20,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        reg_lambda=1.0,
        objective="binary",
        n_jobs=1,
        verbosity=-1,
        random_state=42,
    )
