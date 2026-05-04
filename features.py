"""Feature engineering for variant-triage. EDITABLE.

Starter is identity — the 12 columns from data.py are passed through
as-is. The agent is welcome to add:

- Interaction terms (revel × gnomad_af, charge_delta × hydropathy_delta).
- Polynomial expansions (revel², revel × codon_pos).
- Non-linear warps (log of |af|, rank-transformed scores).
- Derived columns from the biochemical tables (Grantham distance,
  BLOSUM62 substitution score, hydrophobic moment shift).
- Row-level aggregates (sum / mean of the numeric columns, z-score vs
  the train marginals).

Stay under the 4 GB memory cap — float32 + chunked transforms help.
A few dozen derived columns × ~200 K rows × 4 B ≈ 30 MB per column;
plenty of room.
"""
from __future__ import annotations

import numpy as np

from data import FEATURE_COLS

GENE_ID_COL_INDEX = FEATURE_COLS.index("gene_id")
CHROM_ID_COL_INDEX = FEATURE_COLS.index("chrom_id")
CONSEQUENCE_COL_INDEX = FEATURE_COLS.index("consequence_id")

# Indexes that LightGBM should treat as categorical when fitting.
CAT_FEATURE_INDICES = (GENE_ID_COL_INDEX, CHROM_ID_COL_INDEX, CONSEQUENCE_COL_INDEX)


def transform(X: np.ndarray) -> np.ndarray:
    """Identity baseline (REVEL + gene_id columns are already in X)."""
    return X.astype(np.float32, copy=False)
