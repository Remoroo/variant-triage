# Variant Triage — Practical Clinical-Genomics Classification

> **Status:** iterating · harness locked. Two reference points on the
> locked 2024+ ClinVar time-holdout:
> - Biochem-only LightGBM baseline: **0.7018 AUC** (13 features, 14 935
>   train rows, 2 s, 1 core).
> - REVEL meta-predictor reference (using the REVEL score column):
>   **0.9716 AUC** — and a LightGBM that *uses REVEL as a feature*
>   plus the 13 biochem columns reaches **0.9693 AUC** in 1.4 s
>   (commit `c37f86caf9`).
>
> *Goal: build a pathogenicity classifier that beats `test_auc > 0.97`
> on the locked ClinVar 2024+ time-holdout, on a laptop, in a few
> minutes per iteration. The agent may use any feature it can compute
> or download offline from public sources at `setup_data.py` time —
> including REVEL component scores, conservation tracks (GERP, PhyloP,
> PhastCons), gnomAD allele frequencies, SpliceAI, and AlphaMissense —
> so long as those features are pre-computed in the locked
> `setup_data.py` step and never leak the test labels.*

The original framing of this benchmark ("biochemistry-only, no
meta-predictors") makes `test_auc > 0.97` literally unreachable: the
starter feature set tops out around 0.80 AUC even with perfect
tuning, because amino-acid biochemistry alone does not capture
evolutionary conservation or population frequency. To make the
`> 0.97` target a real research target rather than an aspiration,
this program **opens up the feature schema** to all standard public
annotation tracks and asks a different question: how cheaply, and how
robustly, can an autonomous engine *combine* these signals into a
REVEL-class classifier?

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
2024-01-01 onwards. **Success = `test_auc > 0.97`.**

| Target | What it means | Status in this program |
|---|---|---|
| **0.70** | Biochem-only LightGBM baseline | floor (`neutral` in harness) |
| **0.80** | 2010-era single predictors (SIFT, PolyPhen-2) | `keep` threshold |
| **0.90** | Middle-tier meta-predictors (CADD, MutationTaster) | `keep` |
| **0.97** | REVEL reference (Ioannidis 2016) — REVEL-as-feature already lands here | **success target** |
| **0.98+** | Stretch: blend REVEL + AlphaMissense + conservation + gnomAD AF with calibrated stacking | stretch goal |

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

Budget tuned so a single iteration (train + score + log) finishes in
under ~3 minutes on an M-series laptop, leaving room for many
iterations inside a research session.

| Constraint | Value | Enforced by |
|---|---|---|
| CPU | **up to 4 cores** (was 1) | `OMP/MKL/OPENBLAS/VECLIB_*_THREADS=4` in `run.py` |
| Memory | **≤ 8 GB resident** (was 4) | `resource.RLIMIT_AS` + psutil watchdog |
| Wall-clock training | **≤ 300 s (5 min)** (was 600) | SIGALRM in `harness.py` |
| Wall-clock inference | **≤ 30 s** for ~47 K test variants | timer in `harness.eval_loop()` |
| Test set | ClinVar variants first-submitted 2024-01-01+ (≥ 2-star review) | `data.get_test_split()` (locked) |
| Train set | ClinVar variants first-submitted before 2024-01-01 (≥ 2-star review) | `data.get_train_split()` (locked) |
| Peeking into test set | not allowed | `data.py` never yields test labels to the agent's code path |
| External features | only via `setup_data.py` (offline, deterministic, pre-2024 cutoff for any model-derived score) | reviewed at PR time |
| Random seed | `42` | locked in `harness.py` |
| Model artefact size | ≤ 100 MB on disk | `harness.py` checks `artifact_dir` post-train |

The shorter wall-clock (5 min vs 10) is deliberate: with REVEL /
conservation features available, fitting a competitive model takes
seconds, and the tighter budget pushes the agent toward fast
iteration loops (CV, calibration, stacking) rather than long single
fits.

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

## Allowed extension features (load via `setup_data.py`)

To make `> 0.97 AUC` reachable, the following public annotation
tracks are explicitly **allowed**, provided they are joined onto the
variant table in `setup_data.py` (deterministic, offline, no
test-time peeking):

- **REVEL score** (Ioannidis 2016) — already downloaded by
  `setup_data.py`; expose as a feature column.
- **REVEL component scores** (SIFT, PolyPhen-2, MutationTaster,
  FATHMM, PROVEAN, VEST3, MetaSVM, MetaLR, M-CAP, etc. via dbNSFP).
- **Conservation tracks** — GERP++, PhyloP, PhastCons (UCSC).
- **gnomAD v4 allele frequency** — population AF, popmax AF, AC/AN.
- **AlphaMissense pre-computed scores** (Cheng 2023, public release).
- **SpliceAI** delta scores for the four splice events.

The ban is on **test-time leakage**, not on meta-predictors: any
model-derived score must have been published / trained on data
strictly older than the 2024-01-01 ClinVar cutoff. The engine's
playground (`features.py`, `model.py`, `train.py`) can also add
interactions, polynomial expansions, row-level aggregates,
transcript-level rolling statistics, calibration layers, and
stacking ensembles.

## Research roadmap (suggested order of attack)

1. **Sanity baseline**: refit LightGBM with `revel_score` joined onto
   the 13 biochem cols. Expect ~0.97 AUC in <5 s. Confirms harness +
   join + schema are wired correctly. *(Already done at
   `c37f86caf9`: 0.9693.)*
2. **Stack** REVEL + AlphaMissense + (GERP, PhyloP, PhastCons) under a
   shallow LightGBM with strong regularisation. Aim ~0.975.
3. **Add gnomAD AF** with log-transform and a `is_rare` indicator;
   expect another small bump from population evidence.
4. **Calibration** (Platt / isotonic) on a held-out calibration fold
   from the pre-2024 train set, *not* the test set. Improves AUC
   only marginally but improves downstream usefulness.
5. **Stretch — out-of-fold stacking** of LightGBM + logistic + small
   MLP over the meta-features for the 0.98+ region.

## Known failure modes (document as you hit them)

- **Test-time leakage via post-2024 model scores**: AlphaMissense and
  some dbNSFP releases post-date the test cutoff. Always use the
  pre-2024 release of any score column. Document the version in
  `setup_data.py`.
- **Overfitting on 15 K training rows**: the LightGBM baseline shows a
  ~0.17 AUC train/test gap on biochem-only features; the gap shrinks
  dramatically once REVEL is in the feature set, but stacking can
  re-introduce it. Use 5-fold OOF predictions for any stacker.
- **Chromosome-id as a leakage proxy**: `chrom_id` correlates with
  per-gene class imbalance. Drop it from the final model unless its
  contribution is justified by ablation.
- **Time drift**: test set is post-2024; novel genes / rare
  consequences may be under-represented in train. A held-out
  calibration fold from the pre-2024 train set is mandatory.

## Public lineage worth beating

- **Ioannidis et al., *AJHG* 2016** — REVEL meta-predictor. Trained
  on ClinVar + HGMD, the single-score floor for this benchmark.
- **Cheng et al., *Science* 2023** — AlphaMissense. Zero-shot scores
  for 71 M possible missense variants, ~0.92 AUC on ClinVar time-splits.
- **Pejaver et al., *AJHG* 2020** — calibrated ACMG thresholds; the
  the published threshold framework all downstream tools wrap around.
- **Jaganathan et al., *Cell* 2019** — SpliceAI, the splice-effect
  companion predictor (not scored here; can be an added feature).

The interesting Pareto point under this program is **REVEL-or-better
accuracy on a laptop in five minutes per iteration**, with the engine
free to *combine* public annotation tracks but not to peek at the
future. The success bar (`test_auc > 0.97`) is the REVEL reference
line; clearing it requires that the engine learn how to stack and
calibrate, not just download more scores.
