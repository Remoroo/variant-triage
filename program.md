# Variant Triage — Constrained Clinical-Genomics Classification

> **Status:** iterating · harness locked, biochem-only baseline measured
> at **0.7018 AUC** (13 features, 14 935 train rows, 2 s LightGBM on 1
> core). REVEL-alone reference on the same held-out split: **0.9716 AUC**.
>
> *Given a human missense variant and only **raw biochemistry + codon
> position** (hydropathy, charge, Grantham distance, BLOSUM62, genomic
> position), rediscover REVEL-class pathogenicity prediction on a
> ClinVar time-holdout test set. No REVEL, no gnomAD, no conservation
> scores, no meta-predictor features — just what you can compute from
> the AA substitution and its position. Single CPU core, 4 GB RAM,
> 10 minutes of training time on an Apple Silicon Mac.*

This is the classical-ML variant-classification problem stripped to
first principles. REVEL (Ioannidis 2016) is the reference ceiling
because it already ensembles 13 component scores trained on hundreds
of thousands of curated variants. AlphaMissense (Cheng 2023) beats it
because it fine-tunes a protein-language model on Google's TPU pods.
This benchmark asks a different question: starting from the 20×20
substitution matrix and one CPU core, how close to REVEL-class can an
autonomous engine get in ten minutes?

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

ROC AUC against the 47 254 held-out ClinVar variants last-evaluated
2024-01-01 onwards. Higher = better.

| Target | What it means |
|---|---|
| **0.70** | Biochem-only LightGBM baseline (the floor the engine starts from) |
| **0.80** | 2010-era single predictors (SIFT ~0.82, PolyPhen-2 ~0.85 on similar splits) — marks `keep` in the harness |
| **0.90** | Middle-tier meta-predictors (MutationTaster, CADD on rare variants) |
| **0.97** | REVEL reference ceiling (Ioannidis 2016) — uses REVEL's ensemble directly; unreachable without a meta-predictor feature |

Published reference lineage (for context; the engine competes with the
*quality* of these predictors, not by using their scores):

- **REVEL** (Ioannidis et al., *AJHG* 2016) — ensemble of 13 component
  scores, 0.9716 AUC on this time-holdout.
- **AlphaMissense** (Cheng et al., *Science* 2023) — zero-shot PLM-
  derived scores, ~0.92-0.94 AUC on ClinVar time-splits.
- **SIFT** (Kumar 2009), **PolyPhen-2** (Adzhubei 2010) — the classical
  single-score predictors at the 0.80 line.

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

## Starter feature set (13 columns, no meta-predictors)

Included out of the box via `features.transform()`:

Substitution biochemistry (9 cols):

- **`aa_from_hydropathy`, `aa_to_hydropathy`, `hydropathy_delta`** —
  Kyte-Doolittle hydropathy index for source + target amino acids.
- **`aa_from_charge`, `aa_to_charge`, `charge_delta`** — integer charge
  at physiological pH (−1, 0, +1).
- **`aa_volume_delta`** — Grantham-style AA volume difference (Å³).
- **`grantham_distance`** — Grantham (1974) combined composition /
  polarity / volume distance (0-215).
- **`blosum62_score`** — log-odds of this substitution in conserved
  protein alignments (Henikoff & Henikoff 1992).

Positional / genomic context (3 cols):

- **`codon_pos`** — 0 / 1 / 2 codon position of the substituted base.
- **`chrom_id`** — integer chromosome id (1-22 + X=23 + Y=24 + MT=25).
- **`pos_mod1000`** — genomic position modulo 1000, a cheap proxy for
  CpG-island / local-context effects.

Reserved (1 col, always 0 today):

- **`consequence_id`** — molecular consequence id. Only missense
  variants pass the ClinVar filter today; the column is kept so future
  extensions to `stop_gained` / `inframe_del` don't need a schema bump.

REVEL, gnomAD frequencies, conservation scores (GERP, PhyloP,
PhastCons), SpliceAI outputs and other meta-predictor scores are
**deliberately omitted** from the schema — they would collapse the
benchmark to a one-liner (REVEL alone already scores 0.9716 AUC).
The engine's playground (`features.py`) can add interactions,
polynomial expansions, row-level aggregates, transcript-level rolling
statistics — but cannot add any new external score source.

## Known failure modes (document as you hit them)

- **Overfitting on 15 K training rows**: with only 14 935 train
  variants and a 47 254-row test split, the LightGBM baseline already
  shows a 0.17 AUC train/test gap. Aggressive regularisation (fewer
  leaves, larger `min_data_in_leaf`, stronger bagging) or Bayesian
  boosting with shrinkage should help. Out-of-fold predictions via
  k-fold CV are a good alternative to the static 90/10 split the
  baseline uses.
- **Chromosome-id as a leakage proxy**: `chrom_id` is a valid signal
  (some chromosomes are more disease-enriched in ClinVar) but it
  *also* correlates with per-gene class imbalance. Treat with
  scepticism; monitor whether `chrom_id` features dominate feature
  importance at the expense of biochemistry.
- **No meta-predictor features**: the benchmark enforces this via the
  parquet schema. Trying to reintroduce REVEL / CADD / SpliceAI via
  `features.py` is cheating. The engine's playground only gets the 13
  primitives from `FEATURE_COLS`.
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

The interesting Pareto point is **REVEL-class accuracy from biochemistry
alone, on a laptop in ten minutes** — no GPU, no foundation model, no
REVEL-as-a-feature shortcut. The ceiling is real; REVEL already did
the hard work in 2016. The question is how much of that work the
engine can reconstruct from first principles in a constrained loop.
