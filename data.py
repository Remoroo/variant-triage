"""Locked variant-triage data loaders. DO NOT EDIT.

Canonical time-holdout split:
    - train: ClinVar variants first-submitted <  2024-01-01
    - test : ClinVar variants first-submitted >= 2024-01-01
    Both restricted to:
      - missense variants only (molecular consequence == missense)
      - ≥ 2-star review status (criteria_provided_multiple_submitters_no_conflicts
        OR reviewed_by_expert_panel)
      - clinical_significance ∈ {Pathogenic, Likely_pathogenic, Benign, Likely_benign}
      - a REVEL score is available (drop variants with no meta-predictor score)

Reads from `data/splits/{train,test}.parquet` produced by setup_data.py.
The agent never directly imports `get_test_split()` for test-set labels;
only the harness does. `get_train_split()` is fair game.

Feature schema is frozen: if you want more features, derive them in
features.py from these columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SPLITS_DIR = Path(__file__).parent / "data" / "splits"

LABEL_COL = "label"
FEATURE_COLS = (
    "revel",
    "gnomad_af",
    "gnomad_popmax_af",
    "consequence_id",
    "codon_pos",
    "aa_from_hydropathy",
    "aa_to_hydropathy",
    "hydropathy_delta",
    "aa_from_charge",
    "aa_to_charge",
    "charge_delta",
    "aa_volume_delta",
)


def _read_split(which: str) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    path = SPLITS_DIR / f"{which}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python setup_data.py` first"
        )
    table = pq.read_table(str(path))
    y = table.column(LABEL_COL).to_numpy(zero_copy_only=False).astype(np.int8)
    X = np.empty((table.num_rows, len(FEATURE_COLS)), dtype=np.float32)
    for j, col in enumerate(FEATURE_COLS):
        X[:, j] = table.column(col).to_numpy(zero_copy_only=False).astype(np.float32)
    return X, y


def get_train_split(
    rows: int | None = None, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X_train, y_train). Optionally subsample to `rows`
    rows (deterministic with `seed`)."""
    X_tr, y_tr = _read_split("train")
    if rows is None or rows >= len(X_tr):
        return X_tr, y_tr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), size=rows, replace=False)
    idx.sort()
    return X_tr[idx], y_tr[idx]


def get_test_split() -> tuple[np.ndarray, np.ndarray]:
    """LOCKED. Returns the ClinVar 2024-01-01+ time-holdout split."""
    return _read_split("test")
