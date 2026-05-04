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

from features import transform, GENE_ID_COL_INDEX, CAT_FEATURE_INDICES
from metrics import auc
from model import build_model


class _GenePriorWrapper:
    """Wraps a fitted LightGBM that consumes [features..., gene_prior].

    `gene_prior` is a leakage-safe gene-level mean label:
      - at train time: 5-fold OOF mean label per gene (smoothed by global rate).
      - at inference: full-train smoothed mean label per gene.
      - unseen genes get the global prior.
    """

    def __init__(self, base, gene_to_prior, global_prior, gene_col):
        self.base = base
        self.gene_to_prior = gene_to_prior
        self.global_prior = float(global_prior)
        self.gene_col = int(gene_col)

    def _augment(self, X):
        gene = X[:, self.gene_col].astype(np.int64)
        prior = np.full(len(X), self.global_prior, dtype=np.float32)
        # vectorised lookup
        for i, g in enumerate(gene):
            v = self.gene_to_prior.get(int(g))
            if v is not None:
                prior[i] = v
        return np.concatenate([X, prior.reshape(-1, 1)], axis=1)

    def predict_proba(self, X):
        Xt = transform(X)
        Xa = self._augment(Xt)
        return self.base.predict_proba(Xa)

    def predict(self, X):
        return self.predict_proba(X)[:, 1]

    @property
    def booster_(self):
        return self.base.booster_


def _gene_prior_table(genes: np.ndarray, y: np.ndarray, alpha: float = 10.0):
    global_p = float(y.mean())
    out: dict[int, float] = {}
    # group sums via numpy
    order = np.argsort(genes, kind="stable")
    g_sorted = genes[order]
    y_sorted = y[order]
    # find unique gene boundaries
    uniq, starts = np.unique(g_sorted, return_index=True)
    starts = np.append(starts, len(g_sorted))
    for i, g in enumerate(uniq):
        s, e = starts[i], starts[i + 1]
        cnt = e - s
        pos = int(y_sorted[s:e].sum())
        out[int(g)] = (pos + alpha * global_p) / (cnt + alpha)
    return out, global_p


def train_one_run(
    get_train_split: Callable[..., tuple[np.ndarray, np.ndarray]],
    seed: int,
    max_seconds: int,
    artifact_dir: Path,
) -> tuple[Any, float, int]:
    t0 = time.monotonic()

    X, y = get_train_split(seed=seed)
    Xt = transform(X)
    n = len(Xt)

    gene_col = GENE_ID_COL_INDEX
    genes = Xt[:, gene_col].astype(np.int64)

    # 5-fold OOF gene prior to avoid label leakage in train features.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, 5)
    oof_prior = np.empty(n, dtype=np.float32)
    for k, val_idx in enumerate(folds):
        tr_idx = np.concatenate([folds[j] for j in range(5) if j != k])
        table, gp = _gene_prior_table(genes[tr_idx], y[tr_idx])
        for i in val_idx:
            oof_prior[i] = table.get(int(genes[i]), gp)

    # Carve a stratified-ish 10 % held-out for early stopping.
    val_n = max(1500, n // 10)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]

    Xtr_aug = np.concatenate([Xt[tr_idx], oof_prior[tr_idx].reshape(-1, 1)], axis=1)
    Xval_aug = np.concatenate([Xt[val_idx], oof_prior[val_idx].reshape(-1, 1)], axis=1)

    model = build_model()
    elapsed = time.monotonic() - t0
    remaining = max(int(max_seconds - elapsed - 5), 30)
    print(f"[train] rows_train={len(tr_idx):,} rows_val={len(val_idx):,} "
          f"budget_left={remaining}s n_features={Xtr_aug.shape[1]}")

    try:
        import lightgbm as lgb
        model.fit(
            Xtr_aug, y[tr_idx],
            eval_set=[(Xval_aug, y[val_idx])],
            eval_metric="auc",
            categorical_feature=list(CAT_FEATURE_INDICES),
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    except Exception as e:
        print(f"[train] early-stopping path failed: {e!r}; refitting plain")
        model.fit(Xtr_aug, y[tr_idx])

    # Refit gene-prior table on the FULL train set for inference time.
    full_table, full_gp = _gene_prior_table(genes, y)
    wrapped = _GenePriorWrapper(model, full_table, full_gp, gene_col)

    proba = model.predict_proba(Xtr_aug)[:, 1]
    train_auc = auc(y[tr_idx], proba)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.booster_.save_model(str(artifact_dir / "model.txt"))
    except Exception:
        import joblib
        joblib.dump(model, artifact_dir / "model.joblib")

    return wrapped, train_auc, n
