"""Locked harness for variant-triage. DO NOT EDIT.

Enforces:
- 1 CPU core (BLAS / OMP env pinned in run.py before any imports).
- 4 GB RSS cap (resource.RLIMIT_AS + psutil watchdog).
- 600 s training wall-clock budget (SIGALRM).
- 60 s inference budget on the locked test split.
- 100 MB on-disk artefact ceiling.
- Locked ClinVar 2024+ time-holdout test split.

Writes one row per `python run.py` to results.tsv. Crashes and budget
violations are recorded — never silently dropped.
"""
from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from data import get_test_split, get_train_split
from metrics import auc, peak_rss_mb

# ---------- LOCKED CONSTANTS ------------------------------------------------
SEED = 42
WALL_CLOCK_BUDGET_S = 600
INFERENCE_BUDGET_S = 60
MEM_CAP_MB = 4 * 1024
ARTIFACT_CAP_MB = 100
RESULTS = Path(__file__).parent / "results.tsv"
ARTIFACTS = Path(__file__).parent / "artifacts"
HEADER = (
    "commit\ttest_auc\ttrain_auc\twall_clock_s\tinference_s\tpeak_rss_mb"
    "\trows_seen_train\tstatus\tdescription"
)
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "uncommitted"


def _git_subject() -> str:
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()[:160]
    except Exception:
        return "uncommitted"


def _ensure_header() -> None:
    if not RESULTS.exists() or RESULTS.read_text().strip() == "":
        RESULTS.write_text(HEADER + "\n")


def _append(row: dict[str, Any]) -> None:
    _ensure_header()
    line = "\t".join(
        str(row.get(c, ""))
        for c in [
            "commit", "test_auc", "train_auc", "wall_clock_s",
            "inference_s", "peak_rss_mb", "rows_seen_train",
            "status", "description",
        ]
    )
    with RESULTS.open("a") as f:
        f.write(line + "\n")


def _set_mem_cap() -> None:
    """Best-effort 4 GiB virtual-memory cap. macOS' RLIMIT_AS is
    advisory for some allocators, so a psutil watchdog backs it up."""
    try:
        cap = MEM_CAP_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except (ValueError, OSError):
        pass


def _start_mem_watchdog(stop_event: threading.Event,
                       peak_box: list[float]) -> threading.Thread:
    def loop():
        while not stop_event.is_set():
            mb = peak_rss_mb()
            if mb > peak_box[0]:
                peak_box[0] = mb
            if mb > MEM_CAP_MB:
                print(f"[harness] RSS {mb:.0f} MB > {MEM_CAP_MB} MB — killing")
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.5)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


class _TimeBudget:
    def __init__(self, seconds: int):
        self.seconds = int(seconds)

    def __enter__(self):
        self._start = time.monotonic()
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._raise)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    def _raise(self, *_):
        raise TimeoutError(f"training exceeded {self.seconds}s budget")


def _artifact_size_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)


def run() -> bool:
    _set_mem_cap()
    np.random.seed(SEED)

    commit = _git_commit()
    description = _git_subject()
    artifact_dir = ARTIFACTS / commit
    artifact_dir.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "commit": commit,
        "description": description,
        "status": "crash",
        "test_auc": "",
        "train_auc": "",
        "wall_clock_s": "",
        "inference_s": "",
        "peak_rss_mb": "",
        "rows_seen_train": "",
    }

    stop = threading.Event()
    peak_box = [0.0]
    _start_mem_watchdog(stop, peak_box)

    try:
        from train import train_one_run

        X_te, y_te = get_test_split()
        t0 = time.monotonic()

        try:
            with _TimeBudget(WALL_CLOCK_BUDGET_S):
                model, train_auc_, rows_seen = train_one_run(
                    get_train_split=get_train_split,
                    seed=SEED,
                    max_seconds=WALL_CLOCK_BUDGET_S,
                    artifact_dir=artifact_dir,
                )
        except TimeoutError:
            row["status"] = "time_exceeded"
            row["wall_clock_s"] = round(time.monotonic() - t0, 2)
            row["peak_rss_mb"] = round(peak_box[0], 1)
            print(f"[harness] training exceeded {WALL_CLOCK_BUDGET_S}s budget")
            _append(row)
            return False
        except MemoryError:
            row["status"] = "mem_exceeded"
            row["wall_clock_s"] = round(time.monotonic() - t0, 2)
            row["peak_rss_mb"] = round(peak_box[0], 1)
            _append(row)
            return False

        wall = time.monotonic() - t0
        row["wall_clock_s"] = round(wall, 2)
        row["train_auc"] = round(float(train_auc_), 4)
        row["rows_seen_train"] = int(rows_seen)

        # Artefact size check
        size_mb = _artifact_size_mb(artifact_dir)
        if size_mb > ARTIFACT_CAP_MB:
            row["status"] = "artifact_too_large"
            row["peak_rss_mb"] = round(peak_box[0], 1)
            print(f"[harness] artefact dir is {size_mb:.1f} MB > {ARTIFACT_CAP_MB} MB")
            _append(row)
            return False

        # Eval (locked test set)
        eval_t0 = time.monotonic()
        scores = (
            model.predict_proba(X_te) if hasattr(model, "predict_proba")
            else model.predict(X_te)
        )
        if scores.ndim == 2 and scores.shape[1] == 2:
            scores = scores[:, 1]
        eval_wall = time.monotonic() - eval_t0
        row["inference_s"] = round(eval_wall, 2)

        if eval_wall > INFERENCE_BUDGET_S:
            row["status"] = "inference_too_slow"
            row["peak_rss_mb"] = round(peak_box[0], 1)
            print(f"[harness] inference took {eval_wall:.1f}s > {INFERENCE_BUDGET_S}s")
            _append(row)
            return False

        score = auc(y_te, scores.astype(np.float64))
        row["test_auc"] = round(float(score), 4)
        row["peak_rss_mb"] = round(peak_box[0], 1)
        # Threshold ~ 2010-era single-predictor AUC (SIFT/PolyPhen-2).
        # Biochem-only LightGBM baseline measures ~0.70 on this split.
        row["status"] = "keep" if score >= 0.80 else "neutral"
        print(json.dumps(row, indent=2))
        _append(row)
        return True

    except Exception as e:
        row["status"] = "crash"
        row["description"] = (description + " | " + repr(e))[:160]
        row["peak_rss_mb"] = round(peak_box[0], 1)
        traceback.print_exc()
        _append(row)
        return False
    finally:
        stop.set()
