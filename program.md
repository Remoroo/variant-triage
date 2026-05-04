# Variant Triage — Constrained Clinical-Genomics Classification

> **Status:** iterating · harness locked, baseline pending overnight run.
>
> *Given a human missense variant + standard annotations (REVEL, gnomAD,
> conservation, protein context), predict pathogenicity on a ClinVar
> time-holdout test set. Beat the **REVEL-alone floor (AUC ≈ 0.85)** and
> push toward **AlphaMissense-class performance (AUC ≥ 0.92)** — using
> **a single CPU core, 4 GB of RAM, and 10 minutes of training time** on
> an Apple Silicon Mac.*

This benchmark is the classical-ML core of the variant-classification
problem: same input schema commercial tools like
[Deriva](https://deriva.ai) use as features (population frequency,
meta-predictor scores, protein context), but scored against a **locked
time-holdout split** so the engine can't peek into future ClinVar
submissions.

## Single entry point

```
python setup_data.py   # one-time: download ClinVar + REVEL, time-split
python run.py
```

`run.py` invokes `harness.py`, which:

- pins the process to **a single CPU core** (`OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`
  — all set by `run.py` *before* numpy / lightgbm import),
- imposes a **4 GB resident-memory cap** (`resource.RLIMIT_AS` on POSIX
  + psutil watchdog thread that SIGTERMs over the line),
- imposes a **600 s wall-clock training budget** (SIGALRM),
- loads the **locked 2024-01-01+ ClinVar test split**,
- calls the agent's `train.train_one_run()`,
- scores ROC AUC on the locked test split,
- appends one row to `results.tsv`.

## Headline metric — `test_auc` on LOCKED ClinVar 2024+ split

ROC AUC against the held-out ClinVar variants submitted
2024-01-01 onwards. Higher = better.

| Target | Source / reference |
|---|---|
| **0.85** | REVEL alone (Ioannidis 2016) on 2024+ time-split (degraded vs published 0.89 because of time-drift) |
| **0.92** | AlphaMissense (Cheng et al., *Science* 2023) — approx. time-split performance |
| **0.95** | Stretch — top-tier ensembles + population frequency priors |

Cross-checks (do not inform the optimiser):

- `train_auc` — overfit signal.
- `wall_clock_s` — must be ≤ 600.
- `peak_rss_mb` — must be ≤ 4096.
- `inference_s` — test-set scoring must be ≤ 60 s.
- `rows_seen_train` — how many variants the agent's recipe actually used.

## Locked constraints (anti-gaming)

| Constraint | Value | Enforced by |
|---|---|---|
| CPU | **1 core** | `OMP/MKL/OPENBLAS/VECLIB_*_THREADS=1` set by `run.py` |
| Memory | **≤ 4 GB resident** | `resource.RLIMIT_AS` + psutil watchdog |
| Wall-clock training | **≤ 600 s (10 min)** | SIGALRM in `harness.py` |
| Wall-clock inference | **≤ 60 s** for ~10 K test variants | timer in `harness.eval_loop()` |
| Test set | ClinVar variants first-submitted 2024-01-01+ (≥ 2-star review) | `data.get_test_split()` (locked) |
| Train set | ClinVar variants first-submitted before 2024-01-01 (≥ 2-star review) | `data.get_train_split()` (locked) |
| Peeking into test set | not allowed | `data.py` never yields test labels to the agent's code path |
| Random seed | `42` | locked in `harness.py` |
| Model artefact size | ≤ 100 MB on disk | `harness.py` checks `artifact_dir` post-train |

The harness writes one of
`status ∈ {keep, regress, neutral, crash, time_exceeded, mem_exceeded,
inference_too_slow, artifact_too_large}` for every commit. Missing
rows are bugs.

## Repo layout

| File | Locked? | Purpose |
|---|---|---|
| `run.py` | yes | single entry point |
| `harness.py` | yes | CPU pin, mem cap, time budget, locked eval, results.tsv |
| `data.py` | yes (loader, split, schema) | reads pre-built train/test parquet |
| `metrics.py` | yes | ROC AUC, peak RSS reader |
| `setup_data.py` | yes | one-time download + join + time-split |
| `features.py` | no | the agent's playground (engineering) |
| `model.py` | no | the agent's playground (booster / MLP / blends) |
| `train.py` | no | the agent's playground (CV, early stopping, stacking) |

## Starter feature set (in the agent's playground)

Included out of the box via `features.transform()`:

- **`revel`** — REVEL meta-predictor score (2016), 0-1 pathogenicity.
- **`gnomad_af`** — gnomAD v4 allele frequency (global), −1 if absent.
- **`gnomad_popmax_af`** — gnomAD v4 popmax AF, −1 if absent.
- **`consequence_id`** — integer-encoded molecular consequence
  (missense, stop_gained, start_lost, inframe_del, ...).
- **`codon_pos`** — 0 / 1 / 2 codon position of the substituted base.
- **`aa_from_hydropathy`, `aa_to_hydropathy`, `hydropathy_delta`** —
  Kyte-Doolittle scale for source + target amino acids + the delta.
- **`aa_from_charge`, `aa_to_charge`, `charge_delta`** — integer charge
  at physiological pH (-1, 0, +1).
- **`aa_volume_delta`** — Grantham-style AA volume difference.

All real-valued, pre-normalised in `features.transform()`. The agent
can add interactions, polynomial expansions, protein-level rolling
features, log-transforms — whatever it wants — inside `features.py`.

## Known failure modes (document as you hit them)

- **Gene-symbol leakage**: if the agent one-hot-encodes gene symbol,
  it'll overfit to genes seen in train. The time-split helps but
  doesn't eliminate it — new variants in familiar genes are easier.
  Prefer gene-level frequency features (e.g. intolerance Z-score)
  over raw symbols.
- **Review-status leakage**: ClinVar `review_status` correlates with
  label quality. It is NOT included in the features (the harness
  filters it out in `data.py`) — don't try to reintroduce it.
- **Time drift**: test set is post-2024; some novel genes / rare
  consequences may be under-represented in train. A held-out
  calibration fold from the train set is a good idea.

## Public lineage worth beating

- **Ioannidis et al., *AJHG* 2016** — REVEL meta-predictor. Trained
  on ClinVar + HGMD, the single-score floor for this benchmark.
- **Cheng et al., *Science* 2023** — AlphaMissense. Zero-shot scores
  for 71 M possible missense variants, ~0.92 AUC on ClinVar time-splits.
- **Pejaver et al., *AJHG* 2020** — calibrated ACMG thresholds; the
  published pipeline that downstream tools like Deriva wrap around.
- **Jaganathan et al., *Cell* 2019** — SpliceAI, the splice-effect
  companion predictor (not scored here; can be an added feature).

The interesting Pareto point is **AlphaMissense-class accuracy on
a laptop in ten minutes** — no GPU, no foundation-model fine-tuning,
just classical ML on public annotations.
