"""Training recipe for variant-triage. EDITABLE.

The harness calls `train_one_run(get_train_split, seed, max_seconds,
artifact_dir)` and times it. Returns (model, train_auc, rows_seen).

Constraints (harness enforces, but your code plays within them):
- 1 CPU core (BLAS / OMP env already pinned by run.py).
- ≤ 4 GB resident memory.
- ≤ 600 s wall clock (SIGALRM at the budget).
- Locked test split is invisible to you. You may carve a held-out
  chunk of the train rows for early stopping (as the baseline does),
  but never touch the canonical test set.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from features import transform
from metrics import auc
from model import build_model


def train_one_run(
    get_train_split: Callable[..., tuple[np.ndarray, np.ndarray]],
    seed: int,
    max_seconds: int,
    artifact_dir: Path,
) -> tuple[Any, float, int]:
    """Baseline: use all available train rows, 10 % held out for early
    stopping, LightGBM with early_stopping=50 rounds."""
    t0 = time.monotonic()

    X, y = get_train_split(seed=seed)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1000, len(X) // 10)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    X_tr, y_tr = transform(X[tr_idx]), y[tr_idx]
    X_val, y_val = transform(X[val_idx]), y[val_idx]

    model = build_model()

    elapsed = time.monotonic() - t0
    remaining = max(int(max_seconds - elapsed - 5), 30)
    print(f"[train] rows_train={len(X_tr):,} rows_val={len(X_val):,} "
          f"budget_left={remaining}s")

    try:
        import lightgbm as lgb
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    except Exception:
        model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_tr)[:, 1]
    train_auc = auc(y_tr, proba)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.booster_.save_model(str(artifact_dir / "model.txt"))
    except Exception:
        import joblib
        joblib.dump(model, artifact_dir / "model.joblib")

    return model, train_auc, len(X_tr) + len(X_val)
