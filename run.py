"""Variant Triage entry point. DO NOT EDIT.

Single-thread BLAS/OpenMP pinning happens BEFORE numpy/sklearn/lightgbm
are imported — otherwise the pin is a no-op because those libraries
cache thread pool sizes at import time.
"""
from __future__ import annotations

import os
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "1"

from harness import run

if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
