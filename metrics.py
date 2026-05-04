"""Locked metrics for variant-triage. DO NOT EDIT."""
from __future__ import annotations

import numpy as np

# numpy 2.x renamed trapz → trapezoid; fall back for older wheels.
try:
    _trapezoid = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapezoid = np.trapz  # type: ignore[attr-defined]


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC. Implemented locally so we don't depend on sklearn version."""
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score).astype(np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    if tps[-1] == 0 or fps[-1] == 0:
        return float("nan")

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    return float(_trapezoid(tpr, fpr))


def peak_rss_mb(pid: int | None = None) -> float:
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0
